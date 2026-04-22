import os
import torch
import pandas as pd
import time  
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from data_loader import C2MRDataset
from retrieval import CrossModalRetriever
from recipe import RAGPhysicCoT, IngredientConstraintLogitsProcessor
from evaluation import RQ1Evaluator

import warnings
warnings.filterwarnings("ignore")


def init_qwen_vl_model(model_path="/home/student/.cache/modelscope/hub/models/qwen/Qwen-VL-Chat", lora_path="./qwen-vl-lora-c2mr"):

    """
    Initializes the Qwen-VL model and merges LoRA weights if they exist.
    """

    print(f"====== Loading base model from: {model_path} ======")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda:0", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).eval()
    
    if os.path.exists(lora_path):
        print(f"LoRA weights found at {lora_path}. Merging into base model")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("Weight merging complete!")
    else:
        print("No LoRA weights detected. Proceeding with Zero-Shot inference")
        
    try: 
        model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    except: 
        pass
    if not hasattr(model.generation_config, 'chat_format'): 
        model.generation_config.chat_format = 'chatml'
        
    return tokenizer, model

def prepare_chat_input(tokenizer, image_path, text_prompt):

    """
    Formats the input into the ChatML format required by Qwen-VL.
    """

    query = tokenizer.from_list_format([{'image': image_path}, {'text': text_prompt}])
    raw_text = (
        "<|im_start|>system\n"
        "你是一位专业的特级中餐厨师。请根据图片和指令生成精确的食谱。<|im_end|>\n"
        "<|im_start|>user\n"
        f"{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tokenizer(raw_text, return_tensors='pt', padding=False)

def extract_ingredients_from_name(dish_name, all_ingredients):

    """
    Heuristic: Force recall ingredients that appear directly in the dish name.
    """

    return {ing for ing in all_ingredients if len(ing) > 1 and ing in dish_name}

def run_recipe():
    # 1. Initialize model and evaluator
    tokenizer, model = init_qwen_vl_model()
    evaluator = RQ1Evaluator()
    
    # 2. Prepare dataset and Cross-Modal Retriever
    dataset = C2MRDataset("../data/input/C2MR_test.json", image_cache_dir="../image_cache")
    retriever = CrossModalRetriever(K=3)
    retriever.build_index(dataset) 
    
    # 3. Initialize Chain-of-Thought (CoT) framework
    cot_framework = RAGPhysicCoT(model, tokenizer, N=5, tau=0.75) 
    all_results = []
    time_stats = [] 


    gts_dict = {}
    res_dict = {}
    
    test_range = range(len(dataset))
    print(f"\n====== Starting Inference Pipeline, processing {len(test_range)} items ======")

    for i in test_range: 
        _, v_local_path, ref_ingredients, ref_instructions, meta = dataset.get_recipe_data(i)
        
        if v_local_path is None: 
            print(f" Warning: Skipping item {i}, image path is empty.")
            continue
            
        print(f"\n[{i+1}/{len(test_range)}] Processing Dish: {meta['dish_name']}")
        
        # ---------------------------------------------------------
        # # Synchronize GPU for accurate timing
        # ---------------------------------------------------------
        torch.cuda.synchronize()
        start_total = time.perf_counter()
        
        # --- PHASE 1: Retrieval ---
        t1_start = time.perf_counter()
        
        m_prior = retriever.retrieve(v_local_path) 
        
        # Extract a snippet of the ingredients for the prompt
        prior_str = "参考配料库：\n" + "\n".join([f"- {p['ingredients'][:30]}..." for p in m_prior])
        t1_end = time.perf_counter()
        
        # --- PHASE 2: Multi-path Sampling and Entropy-based Filtering ---
        t2_start = time.perf_counter()
        prompt_text = (
            f"{prior_str}\n"
            f"这是{meta['dish_name']}的成品图。\n"
            f"请仔细观察图片内容，结合参考库，提取出这道菜真正用到的食材。\n"
            f"【警告】仅列出食材名称！使用逗号分隔，不要写做法！例如：西红柿, 鸡蛋, 盐"
        )
        inputs = prepare_chat_input(tokenizer, v_local_path, prompt_text).to(model.device)
        
        all_sequences = []
        all_step_scores = []
        with torch.no_grad():
            for n_idx in range(cot_framework.N):
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=True, 
                    top_p=0.8, temperature=0.6, num_return_sequences=1, 
                    output_scores=True, return_dict_in_generate=True
                )
                all_sequences.append(outputs.sequences[0])
                all_step_scores.append(outputs.scores)
                torch.cuda.empty_cache()
        
        entity_uncertainties = cot_framework.calculate_entity_entropy(all_sequences, all_step_scores)
        hat_I = cot_framework.dual_calibration_filtering(entity_uncertainties)
        
        name_ings = extract_ingredients_from_name(meta['dish_name'], dataset.all_ingredients)
        hat_I.update(name_ings)
        if len(hat_I) < 2: hat_I.add(meta['dish_name'][:2])
        t2_end = time.perf_counter()
        
        # --- PHASE 3: Synthesis with Logical Constraints ---
        t3_start = time.perf_counter()
        final_prompt = (
            f"这道菜是{meta['dish_name']}。\n"
            f"你必须使用且仅使用这些食材：{','.join(list(hat_I))}。\n"
            f"请生成详细的烹饪步骤，逻辑清晰，分点描述："
        )
        final_inputs = prepare_chat_input(tokenizer, v_local_path, final_prompt).to(model.device)
        logits_processor = IngredientConstraintLogitsProcessor(tokenizer, hat_I)
        
        with torch.no_grad():
            final_outputs = model.generate(
                **final_inputs, max_new_tokens=512, do_sample=False, 
                logits_processor=[logits_processor], repetition_penalty=1.1
            )
        
        full_response = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
        final_recipe_text = full_response.split("assistant")[-1].strip() if "assistant" in full_response else full_response
        
        # Storage for metrics
        # res_dict
        res_dict[i] = [final_recipe_text]
        # gts_dict
        gts_dict[i] = ["".join(ref_instructions)] 


        torch.cuda.synchronize()
        t3_end = time.perf_counter()
        
        end_total = time.perf_counter()
        
        # ---------------------------------------------------------
        # Log latency
        # ---------------------------------------------------------
        duration_total = end_total - start_total
        dur_retrieval = t1_end - t1_start
        dur_sampling = t2_end - t2_start
        dur_synthesis = t3_end - t3_start
        
        time_info = {
            "total_latency": duration_total,
            "retrieval_ms": dur_retrieval * 1000,
            "sampling_sec": dur_sampling,
            "synthesis_sec": dur_synthesis
        }
        time_stats.append(time_info)
        
        print(f"  Generated Recipe:\n{final_recipe_text[:100]}...\n")
        print(f"  Latency: Total {duration_total:.2f}s [Retrieval: {dur_retrieval*1000:.1f}ms | Sampling/Entropy: {dur_sampling:.2f}s | Synthesis: {dur_synthesis:.2f}s]")
        
        # 4. Evaluation Metrics
        ref_text_full = "".join(ref_instructions)
        metrics = {
            **evaluator.compute_ngram_metrics(ref_text_full, final_recipe_text),
            **evaluator.compute_ingredient_alignment(ref_ingredients, final_recipe_text, dataset.all_ingredients),
            **evaluator.compute_chair_i(ref_ingredients, final_recipe_text, dataset.all_ingredients)
        }
        print(f" Current Metrics: {metrics}")
        all_results.append(metrics)

    # 5. Global summary
    print("\n" + "="*60)
    print("【PERFORMANCE SUMMARY】")
    if all_results:
        metrics_df = pd.DataFrame(all_results)
        
        # Calculate global CIDEr
        final_cider = evaluator.compute_corpus_cider(gts_dict, res_dict)
        
        # Calculate the average of other indicators
        summary = metrics_df.mean(numeric_only=True)
        
        summary['CIDEr'] = final_cider
        
        print(summary.round(3))
        
        # Time consumption statistics
        time_df = pd.DataFrame(time_stats)
        
        print("\n[Performance Analysis]")
        print(f"Average time per recipe: {time_df['total_latency'].mean():.2f} 秒")
    print("="*60)

if __name__ == "__main__":
    run_recipe()
