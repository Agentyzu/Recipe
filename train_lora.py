import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from data_loader import C2MRDataset
import ast

# Disable parallelism to prevent deadlocks during tokenization in multi-process environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Standard CLIP Image Preprocessing Pipeline
# Qwen-VL uses a 448x448 input size with specific normalization constants (from OpenAI CLIP)
qwen_image_processor = transforms.Compose([
    transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), 
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

class QwenVLRecipeDataset(Dataset):
    """
    Custom dataset class for fine-tuning Qwen-VL on Chinese recipes.
    Handles interleaved image-text data and label masking for training.
    """
    
    def __init__(self, c2mr_dataset, tokenizer, max_length=1024):
        self.dataset = c2mr_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch raw recipe data
        _, img_path, ingredients, instructions, meta = self.dataset.get_recipe_data(idx)
        
        # Validate image path existence
        if img_path is None or not os.path.exists(img_path):
            img_path = "" 
        else:
            img_path = os.path.abspath(img_path)
            
        dish_name = meta['dish_name']
        ing_str = ",".join(ingredients)
        ins_str = "".join(instructions)

        # 1. Construct Prompt and Response using Qwen-VL chat format
        # System message defines the model's persona
        prompt_text = f"<|im_start|>system\nYou are a professional Master Chef of Chinese Cuisine.<|im_end|>\n<|im_start|>user\n"
        
        if img_path:
            # Use 'from_list_format' to handle image placeholders correctly in the prompt
            query = self.tokenizer.from_list_format([
                {'image': img_path},
                {'text': f"This dish is {dish_name}. Please observe the image carefully, list the actual ingredients used, and provide detailed cooking instructions."}
            ])
            prompt_text += f"{query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt_text += f"This dish is {dish_name}. Please list the actual ingredients used and provide detailed cooking instructions.<|im_end|>\n<|im_start|>assistant\n"

        # The model's expected response
        response_text = f"Ingredients: {ing_str}\nInstructions: {ins_str}<|im_end|>"
        
        # Combine into a single string for tokenization
        full_text = prompt_text + response_text
        
        # 2. Tokenization and Label Masking
        # We tokenize the full sequence and then mask the prompt part so the model only learns to predict the response
        full_tokens = self.tokenizer(
            full_text, 
            return_tensors='pt', 
            padding=False, 
            truncation=True, 
            max_length=self.max_length
        )
        input_ids = full_tokens.input_ids[0]
        
        # Dynamically locate the start of the Assistant's response to create labels
        target_token_ids = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        
        response_start_idx = -1
        for i in range(len(input_ids) - len(target_token_ids)):
            if input_ids[i : i + len(target_token_ids)].tolist() == target_token_ids:
                curr = i + len(target_token_ids)
                # Skip potential newline tokens (up to 3) after the assistant header
                for _ in range(3):
                    if curr < len(input_ids) and self.tokenizer.decode([input_ids[curr]]).strip() == "":
                        curr += 1
                    else:
                        break
                response_start_idx = curr
                break
        
        # Mask the prompt by setting labels to -100 (standard for PyTorch CrossEntropyLoss to ignore)
        labels = input_ids.clone()
        if response_start_idx != -1:
            labels[:response_start_idx] = -100
        else:
            # Fallback: Mask everything if the start tag isn't found (prevents training on noise)
            labels[:] = -100
            print(f"Warning: Assistant start tag not found at index {idx}!")

        return {
            "input_ids": input_ids,
            "labels": labels
        }

def custom_data_collator(features):
    """
    Standard data collator for padding variable-length sequences within a batch.
    """
    features = [f for f in features if len(f["input_ids"]) > 0]
    input_ids = [f["input_ids"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Qwen's specific padding token ID
    pad_id = 151643 
    
    # Pad sequences to the longest in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_id)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def apply_qwen_visual_monkey_patch(model):
    """
    CRITICAL PATCH: Overrides the default visual encoding logic.
    Qwen-VL's default behavior sometimes fails during training when processing image paths 
    embedded in strings. This patch manually handles image loading, preprocessing, 
    and ensures tensor dtype/device alignment.
    """
    core_model = model.base_model.model if hasattr(model, "base_model") else model
    
    if hasattr(core_model, "transformer") and hasattr(core_model.transformer, "visual"):
        original_encode = core_model.transformer.visual.encode

        def safe_encode(images):
            # If the model receives a list of file paths (strings)
            if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
                processed_images = []
                for path in images:
                    try:
                        clean_path = path.replace("file://", "").strip()
                        img = Image.open(clean_path).convert("RGB")
                        # Standard preprocessing to tensor
                        img_tensor = qwen_image_processor(img) 
                        # Align with model's precision (bfloat16) and device
                        img_tensor = img_tensor.to(dtype=torch.bfloat16, device=core_model.device)
                        processed_images.append(img_tensor)
                    except Exception as e:
                        print(f"Warning: Image load failed for {path}: {e}")
                        # Robustness: use a black placeholder image on failure
                        img_tensor = torch.zeros((3, 448, 448), dtype=torch.bfloat16, device=core_model.device)
                        processed_images.append(img_tensor)
                
                # Stack and feed directly into the Vision Transformer
                images_tensor = torch.stack(processed_images)
                return core_model.transformer.visual(images_tensor)
            
            # Handling pre-processed tensors
            elif isinstance(images, torch.Tensor):
                images = images.to(dtype=torch.bfloat16, device=core_model.device)
                return core_model.transformer.visual(images)
            else:
                return original_encode(images)
            
        core_model.transformer.visual.encode = safe_encode
        print("Visual Pipeline Patch Applied: Manual image interception active.")

def train_lora_model():
    # Configuration paths
    model_dir = "/home/student/.cache/modelscope/hub/models/qwen/Qwen-VL-Chat"
    output_dir = "./qwen-vl-lora-c2mr"
    
    print("====== 1. Initializing Model (BFloat16 Precision) ======")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id
    
    # Load model in bfloat16 for better numerical stability and performance on modern GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="cuda:0", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    print("====== 2. Injecting LoRA Adapters ======")
    # Configure Low-Rank Adaptation (LoRA)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["c_attn", "attn.c_proj", "w1", "w2"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Apply the visual patch to prevent crashes during multimodal training
    apply_qwen_visual_monkey_patch(model)
    
    print("====== 3. Preparing Datasets (Train & Val) ======")
    # Load raw training and validation data
    raw_dataset_train = C2MRDataset("C2MR_train.json", image_cache_dir="./image_cache")
    train_dataset = QwenVLRecipeDataset(raw_dataset_train, tokenizer, max_length=1024)
    
    raw_dataset_val = C2MRDataset("C2MR_val.json", image_cache_dir="./image_cache")
    val_dataset = QwenVLRecipeDataset(raw_dataset_val, tokenizer, max_length=1024)
    
    # Safety Check: Verify that the label masking is working correctly
    test_sample = train_dataset[0]
    valid_len = (test_sample['labels'] != -100).sum().item()
    print(f"\n[Safety Check] Sample 0: Number of tokens to be predicted: {valid_len}")
    if valid_len == 0:
        print("!!! CRITICAL ERROR: Valid label length is 0. Check prompt tags! !!!")
        return
    
    print("====== 4. Starting Trainer Pipeline ======")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        optim="adamw_torch",           
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        
        # Evaluation and Checkpointing Logic
        evaluation_strategy="epoch",     # Evaluate after every epoch
        save_total_limit=2,              # Keep only the last 2 checkpoints
        load_best_model_at_end=True,     # Best weights based on validation metrics
        metric_for_best_model="loss",
        greater_is_better=False,
        
        bf16=True,                       # Use BFloat16 mixed precision
        remove_unused_columns=False,    
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
    )
    
    trainer.train()
    
    print("====== 5. Saving Final Model Weights ======")
    # Save the adapter weights and the configuration
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training Complete! LoRA model saved to {output_dir}")


if __name__ == "__main__":
    train_lora_model()