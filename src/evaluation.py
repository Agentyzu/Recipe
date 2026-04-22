import os
import numpy as np
import re
import jieba
import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score

class ChineseCider:
    """
    Pure Python implementation of the CIDEr (Consensus-based Image Description Evaluation) 
    metric optimized for Chinese text.
    
    Calculates the TF-IDF weighted cosine similarity of n-grams (1-4) between 
    generated text and a set of reference texts.
    
    :param n: Maximum n-gram length to consider (default is 4).
    :param sigma: Sigma parameter for Gaussian smoothing (standard CIDEr parameter).
    """
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma

    def get_ngram_counts(self, words):
        """
        Extracts n-gram counts from a list of words.
        
        :param words: List of tokens.
        :return: List of Counters, where each Counter represents n-gram frequencies for a specific 'n'.
        """
        res = []
        for i in range(1, self.n + 1):
            res.append(Counter(tuple(words[j:j+i]) for j in range(len(words)-i+1)))
        return res

    def compute_score(self, gts_dict, res_dict):
        """
        Computes the final CIDEr score for a corpus.
        
        :param gts_dict: Dictionary mapping sample IDs to a list of reference strings.
        :param res_dict: Dictionary mapping sample IDs to a list containing the generated string.
        :return: Mean CIDEr score across all samples.
        """
        # 1. Tokenization and N-gram Statistics Collection
        all_res_ngrams = []
        all_gts_ngrams = []
        df_stats = [Counter() for _ in range(self.n)]
        
        common_ids = sorted(res_dict.keys())
        for idx in common_ids:
            res_words = list(jieba.cut(res_dict[idx][0]))
            gts_words = [list(jieba.cut(gt)) for gt in gts_dict[idx]]
            
            res_ng = self.get_ngram_counts(res_words)
            gts_ng = [self.get_ngram_counts(gt) for gt in gts_words]
            
            all_res_ngrams.append(res_ng)
            all_gts_ngrams.append(gts_ng)
            
            # Document Frequency (DF) statistics: Count how many documents contain each n-gram
            for ng_set in gts_ng:
                for i in range(self.n):
                    for gram in ng_set[i]:
                        df_stats[i][gram] += 1

        # 2. CIDEr Calculation
        num_docs = len(common_ids)
        scores = []
        
        for i in range(num_docs):
            res_ng = all_res_ngrams[i]
            gts_ng_list = all_gts_ngrams[i]
            
            doc_score = 0
            for n in range(self.n):
                # Calculate TF-IDF weighted cosine similarity for each n-gram level
                tmp_score = 0
                for gts_ng in gts_ng_list:
                    val = 0
                    norm_res = 0
                    norm_gts = 0
                    
                    # Compute similarity across the union of all observed n-grams
                    all_grams = set(res_ng[n].keys()) | set(gts_ng[n].keys())
                    for gram in all_grams:
                        # IDF = log(N / DF)
                        df = df_stats[n].get(gram, 0)
                        idf = math.log(max(1, num_docs / (df + 1e-6)))
                        
                        tf_res = res_ng[n].get(gram, 0)
                        tf_gts = gts_ng[n].get(gram, 0)
                        
                        w_res = tf_res * idf
                        w_gts = tf_gts * idf
                        
                        val += w_res * w_gts
                        norm_res += w_res**2
                        norm_gts += w_gts**2
                    
                    if norm_res > 0 and norm_gts > 0:
                        val /= (math.sqrt(norm_res) * math.sqrt(norm_gts))
                    tmp_score += val
                
                doc_score += (tmp_score / len(gts_ng_list))
            
            scores.append(doc_score / self.n)

        return np.mean(scores) * 10.0

class RQ1Evaluator:
    """
    Evaluation suite for recipe generation models, covering linguistic metrics, 
    ingredient alignment accuracy, and hallucination assessment.
    
    :param safe_list: Common staple ingredients often ignored in hallucination checks.
    """
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.cider_scorer = ChineseCider()
        self.smooth = SmoothingFunction().method1
        self.safe_list = {"水", "油", "盐", "糖", "葱", "姜", "蒜", "酱油", "醋", "料酒", "味精", "鸡精"}

    def compute_corpus_cider(self, gts_dict, res_dict):
        """Calculates CIDEr score for the whole corpus."""
        if not gts_dict or not res_dict: return 0.0
        return round(self.cider_scorer.compute_score(gts_dict, res_dict), 3)

    def compute_ngram_metrics(self, ref_text, gen_text):
        """
        Calculates standard NLP metrics: BLEU-1, BLEU-4, ROUGE-L, and METEOR.
        
        :param ref_text: Ground truth reference text.
        :param gen_text: Model generated text.
        :return: Dictionary containing the four scores.
        """
        ref_tokens = list(ref_text)
        gen_tokens = list(gen_text)

        # Calculate BLEU-1 (Unigram) and BLEU-4 (Cumulative 4-gram)
        bleu1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smooth)
        bleu4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)
        # Calculate ROUGE-L F-measure
        rouge_l = self.rouge_scorer.score(ref_text, gen_text)['rougeL'].fmeasure

        # Calculate METEOR score
        try:
            m_score = meteor_score([ref_tokens], gen_tokens)
        except: m_score = 0.0
        return {"B-1": round(bleu1 * 100, 2), "B-4": round(bleu4 * 100, 2), "R-L": round(rouge_l * 100, 2), "METEOR": round(m_score * 100, 2)}

    def compute_ingredient_alignment(self, ref_ingredients, gen_text, all_dataset_ingredients):
        """
        Domain-specific metric measuring how accurately the model identifies ingredients.
        
        :param ref_ingredients: List of actual ingredients in the recipe.
        :param gen_text: The generated recipe text.
        :param all_dataset_ingredients: Global list of all known ingredients in the dataset.
        :return: IoU, Precision, Recall, and F1 score for ingredient extraction.
        """
        ref_set = set(ref_ingredients)
        mentioned_in_gen = []
        for ing in all_dataset_ingredients:
            if len(ing) > 1 and ing in gen_text and ing not in self.safe_list:
                mentioned_in_gen.append(ing)
        gen_set = set(mentioned_in_gen)
        tp = 0
        # Use substring matching for True Positive calculation (e.g., 'pork' matches 'lean pork')
        for g_ing in gen_set:
            if any(g_ing in r_ing or r_ing in g_ing for r_ing in ref_set): tp += 1
        precision = tp / len(gen_set) if len(gen_set) > 0 else 1.0
        recall = tp / len(ref_set) if len(ref_set) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"IoU": round((tp / (len(ref_set) + len(gen_set) - tp + 1e-6)) * 100, 2), "Prec": round(precision * 100, 2), "Rec": round(recall * 100, 2), "F1": round(f1 * 100, 2)}

    def compute_chair_i(self, ref_ingredients, gen_text, all_dataset_ingredients):
        """
        Calculates the CHAIR_i (Caption Hallucination Assessment with Image Relevance) metric 
        specifically for ingredients.
        
        Measures the percentage of hallucinated ingredients (mentioned in text but not in ground truth).
        
        :param ref_ingredients: Ground truth ingredients.
        :param gen_text: Generated recipe text.
        :return: CHAIR_i percentage (lower is better).
        """
        ref_set = set(ref_ingredients)
        mentioned = [ing for ing in all_dataset_ingredients if len(ing) > 1 and ing in gen_text and ing not in self.safe_list]
        if not mentioned: return {"CHAIR_i": 0.0}
        hallucinated = 0
        for m_ing in mentioned:
            if not any(m_ing in r_ing or r_ing in m_ing for r_ing in ref_set): hallucinated += 1
        return {"CHAIR_i": round(hallucinated / len(mentioned) * 100, 2)}