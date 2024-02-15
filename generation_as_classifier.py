'''
This module implemets ProGen generation as a classifier, comprising two main classes:
1) teacher forcing generation: given a sequence, it computes the resulting generation for eache position in the sequence given all the previous (real) positions. (in addition the probabilities for each amino acid predicted)
2) after-n generation: given a sequence, the model takes the first n aminoacids as prefix and generates until the length of the actual (real) protein in input to the model. (in addition the probabilities for each amino acid predicted)
'''
import torch
import numpy as np
from tokenizer import Tokenizer
from model_manager import VocabularyManager

class GenerationAsClassifier:
    def __init__(self, predict_fn, penalty=0, topk=1):
        vocab_manager = VocabularyManager()
        self.vocab_size = vocab_manager.vocab_size
        self.predict_fn = predict_fn
        self.penalty = penalty
        self.topk = topk
        self.tokenizer = Tokenizer()
        
    def teacher_forcing_generation(self, input_sequence, tax_lineage):
        res = ""
        tokens_prob = []
        key_len = len(tax_lineage) # len(kw_lineage+tax_lineage)
        for i in range(1, len(input_sequence)):
            iteration_input_prefix = input_sequence[:i]
            seed_seq = [self.tokenizer.aa_to_ctrl_idx[ii] for ii in iteration_input_prefix]
            generate_num = key_len + len(seed_seq) + 1 # how many tokens to generate, here only one
            seq_length = min(generate_num, 511)
            text = tax_lineage + seed_seq # tax_lineage + kw_lineage + seed_seq
            padded_text = text + [0] * (generate_num - len(text))
            tokens_generated = np.tile(padded_text, (1,1))
            for token in range(len(text)-1, generate_num-1):
                prompt_logits = self.predict_fn(tokens_generated[:, :seq_length]).squeeze()
                _token = token if token < seq_length else -1
                prompt_logits = prompt_logits.cpu().detach().numpy()
                if self.penalty>0:
                    penalized_so_far = set()
                    # variable token_flag for first amminoacids (to count them if they are less that 4)
                    if token >= key_len + 3:
                        token_flag = 3  
                    elif token - key_len - 3 <= 0:
                        token_flag = 0
                    else:
                        token_flag = token
                    for _ in range(token-token_flag,token+1):
                        generated_token = tokens_generated[0][_] - (self.vocab_size-26) # added
                        if generated_token in penalized_so_far:
                            continue
                        penalized_so_far.add(generated_token)
                        prompt_logits[_token][generated_token] /= self.penalty  
                # compute probabilities from logits
                prompt_probs = np.exp(prompt_logits[_token])
                prompt_probs = prompt_probs / sum(prompt_probs)
                def softmax(x, axis=0):
                    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
                    return e_x / e_x.sum(axis=axis, keepdims=True)
                prompt_probs_test = softmax(prompt_logits[_token])
                pruned_list = np.argsort(prompt_probs)[::-1]
                tokens_prob.append([prompt_probs.tolist()])
                if self.topk==1:
                    idx = pruned_list[0]
                else:
                    pruned_list = pruned_list[:self.topk]
                    chosen_idx = torch.distributions.categorical.Categorical(logits=torch.tensor(np.expand_dims(prompt_logits[_token][pruned_list],0))).sample().numpy()[0]
                    idx = pruned_list[chosen_idx]
                # assign the token for generation
                idx += (self.vocab_size-26) # added to convert 0 AA to original ctrl idx
                tokens_generated[0][token+1] = idx
            tokens_generated_so_far = tokens_generated[0].squeeze()[:token+2]
            tokens_generated_so_far = tokens_generated_so_far[(tokens_generated_so_far>=(self.vocab_size-26)) & (tokens_generated_so_far<(self.vocab_size-1))]
            tokens_generated_so_far = ''.join([self.tokenizer.ctrl_idx_to_aa[c] for c in tokens_generated_so_far])
            query = tokens_generated_so_far[len(seed_seq):]
            res += query
        return res, tokens_prob, 1
    
    def after_n_generation(self, input_sequence, tax_lineage, n):
        # key_len = 0
        res = ""
        tokens_prob = []
        key_len = len(tax_lineage) # len(kw_lineage+tax_lineage)
        i = n
        iteration_input_prefix = input_sequence[:i]
        seed_seq = [self.tokenizer.aa_to_ctrl_idx[ii] for ii in iteration_input_prefix]
        generate_num = key_len + len(seed_seq) + (len(input_sequence) - len(seed_seq)) # how many tokens to generate
        seq_length = min(generate_num, 511)
        text = tax_lineage + seed_seq # tax_lineage + kw_lineage + seed_seq
        padded_text = text + [0] * (generate_num - len(text))
        tokens_generated = np.tile(padded_text, (1,1))
        for token in range(len(text)-1, generate_num-1):
            prompt_logits = self.predict_fn(tokens_generated[:, :seq_length]).squeeze()
            _token = token if token < seq_length else -1
            prompt_logits = prompt_logits.cpu().detach().numpy()
            if self.penalty>0:
                penalized_so_far = set()
                # variable token_flag for first amminoacids (to count them if they are less that 4)
                if token >= key_len + 3:
                    token_flag = 3  
                elif token - key_len - 3 <= 0:
                    token_flag = 0
                else:
                    token_flag = token
                for _ in range(token-token_flag,token+1):
                    generated_token = tokens_generated[0][_] - (self.vocab_size-26) # added
                    if generated_token in penalized_so_far:
                        continue
                    penalized_so_far.add(generated_token)
                    prompt_logits[_token][generated_token] /= self.penalty  
            # compute probabilities from logits
            prompt_probs = np.exp(prompt_logits[_token])
            prompt_probs = prompt_probs / sum(prompt_probs)
            def softmax(x, axis=0):
                e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
                return e_x / e_x.sum(axis=axis, keepdims=True)
            prompt_probs_test = softmax(prompt_logits[_token])
            pruned_list = np.argsort(prompt_probs)[::-1]
            tokens_prob.append([prompt_probs.tolist()])
    
            if self.topk==1:
                idx = pruned_list[0]
            else:
                pruned_list = pruned_list[:self.topk]
                chosen_idx = torch.distributions.categorical.Categorical(logits=torch.tensor(np.expand_dims(prompt_logits[_token][pruned_list],0))).sample().numpy()[0]
                idx = pruned_list[chosen_idx]
            # assign the token for generation
            idx += (self.vocab_size-26) # added to convert 0 AA to original ctrl idx
            tokens_generated[0][token+1] = idx
        tokens_generated = tokens_generated[0][len(seed_seq) + key_len:]
        tokens_generated = ''.join([self.tokenizer.ctrl_idx_to_aa[c] for c in tokens_generated])
        return tokens_generated, tokens_prob, i
