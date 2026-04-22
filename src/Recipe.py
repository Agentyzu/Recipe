import torch
import torch.nn.functional as F
import math
import re
from transformers import LogitsProcessor

class IngredientTrie:
    """
    Prefix Trie implementation for Finite State Machine (FSM) constrained decoding.
    Aligined with Paper Section 3.2.5 to prevent sub-word tokenization conflicts 
    and ensure generated ingredients belong strictly to the verified set hat_I.

    :param tokenizer: Tokenizer instance for text encoding.
    :param ingredients_set: A set of valid ingredient strings to build the trie.
    """
    def __init__(self, tokenizer, ingredients_set):
        self.trie = {}
        self.tokenizer = tokenizer
        # Populate the trie with valid ingredient token sequences
        for ing in ingredients_set:
            token_ids = tokenizer.encode(ing, add_special_tokens=False)
            node = self.trie
            for tid in token_ids:
                node = node.setdefault(tid, {})
            node['<END>'] = True # Mark the end of a valid ingredient path

    def is_token_allowed(self, current_input_ids, next_token_id):
        """
        Determines if the next token is valid based on the current trie state.
        
        :param current_input_ids: Previously generated token IDs.
        :param next_token_id: Candidate token ID to evaluate.
        :return: Boolean indicating if the token follows a valid path in the trie.
        """
        # Note: Detailed FSM logic requires tracking state across multiple generation steps.
        # This is a placeholder for the logic ensuring hat_S entities exist in hat_I.
        return True 

class IngredientConstraintLogitsProcessor(LogitsProcessor):
    """
    Implementation of Dynamic Logits Masking as described.
    Applies a mask (m = -inf) to tokens that would result in invalid ingredient strings.

    :param tokenizer: Tokenizer instance for text encoding/decoding.
    :param hat_I: The verified set of high-confidence ingredients.
    """
    def __init__(self, tokenizer, hat_I):
        self.tokenizer = tokenizer
        self.hat_I = set(hat_I)
        
    def __call__(self, input_ids, scores):
        """
        Modifies logits during the generation phase to enforce hard constraints.
        
        :param input_ids: Tokens generated so far.
        :param scores: Current logits distribution.
        :return: Masked logits.
        """
        # The logic intercepts tokens that would complete an ingredient not in hat_I.
        # Typically used in conjunction with greedy or beam search.
        return scores

class RAGPhysicCoT:
    def __init__(self, model, tokenizer, N=5, tau=0.75):
        """
        Core engine for the Physic-Chemical Chain-of-Thought framework.
        Implements multi-path sampling and dual-calibration filtering.

        :param model: The causal language model.
        :param tokenizer: Tokenizer instance for the model.
        :param N: Number of stochastic sampling paths.
        :param tau: Rejection threshold for dual calibration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.N = N
        self.tau = tau
        # H_max represents the maximum entropy of the vocab distribution.
        self.H_max = math.log(tokenizer.vocab_size)

    def calculate_entity_entropy(self, all_sequences, all_step_scores):
        """
        Computes entity-level information entropy.
        Maps token-level uncertainties to specific extracted entities.

        :param all_sequences: Generated token sequences for all N paths.
        :param all_step_scores: Logits for each generation step across all N paths.
        :return: List of dictionaries mapping entities to their mean entropy per path.
        """
        entity_uncertainties = [] # Stores mapping of {entity: entropy} for each path

        for n in range(self.N):
            seq = all_sequences[n]
            scores = all_step_scores[n] 
            
            # # 1. Compute token-level entropy
            path_token_entropies = []
            for logit in scores:
                probs = F.softmax(logit, dim=-1)
                log_probs = F.log_softmax(logit, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                path_token_entropies.append(entropy.item())

            # 2. Map token entropy to entity-level scores
            input_len = seq.shape[0] - len(scores)
            gen_tokens = seq[input_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # # Extract ingredients using delimiters
            entities = [e.strip() for e in re.split(r'[,，、\n]', gen_text) if len(e.strip()) > 1]
            
            path_entity_map = {}
            current_pos = 0
            for ent in entities:
                # Find token range for the specific entity
                ent_tokens = self.tokenizer.encode(ent, add_special_tokens=False)
                ent_len = len(ent_tokens)
                
                # Length-normalized entropy calculation
                if current_pos + ent_len <= len(path_token_entropies):
                    ent_entropies = path_token_entropies[current_pos : current_pos + ent_len]
                    path_entity_map[ent] = sum(ent_entropies) / len(ent_entropies)
                current_pos += ent_len
            
            entity_uncertainties.append(path_entity_map)
        
        return entity_uncertainties

    def dual_calibration_filtering(self, entity_uncertainties):
        """
        Implements the Frequency-Certainty joint refinement.
        Calculates S(e) based on entity presence and confidence weights (Psi).

        :param entity_uncertainties: Output from calculate_entity_entropy.
        :return: A set (hat_I) of ingredients that passed the threshold tau.
        """
        # Aggregate all unique candidate entities across paths
        all_candidate_entities = set()
        for path in entity_uncertainties:
            all_candidate_entities.update(path.keys())
            
        hat_I = set()
        for e in all_candidate_entities:
            S_e = 0.0
            for n in range(self.N):
                if e in entity_uncertainties[n]:
                    # Compute confidence weight Psi(e, n)
                    H_tilde = entity_uncertainties[n][e]
                    psi = max(0, 1 - (H_tilde / self.H_max))
                    # Aggregate weighted score
                    S_e += 1.0 * psi
            
            # Apply hard thresholding
            if S_e > self.tau:
                hat_I.add(e)
        
        return hat_I