'''
This module implemets ProGen generation as a classifier, comprising three main classes:
1) teacher forcing generation: given a sequence, it computes the resulting generation for eache position in the sequence given all the previous (real) positions. (in addition the probabilities for each amino acid predicted)
2) after-n generation: given a sequence, the model takes the first n aminoacids as prefix and generates until the length of the actual (real) protein in input to the model. (in addition the probabilities for each amino acid predicted)
3) generation_complete_sequence: generates sequences using a fine-tuned model. Stop keyword usage.
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import pickle
import numpy as np
from torch.functional import F
from torch import log
from tokenizer import Tokenizer
from model_manager import VocabularyManager
from model_manager import CTRLmodel

class GeneratorManager:
    def __init__(self, model_path, penalty=0, topk=1, top_p=0.5, temperature=1):
        # let's load our model in memory
        model = CTRLmodel()
        reader = model.loadCheckpoint(model_path=model_path)
        if torch.cuda.is_available():
            model = model.cuda()
            print('GPU aviable. Previous checkpoint loaded in GPU')
        else: 
            print('GPU not aviable. Previous checkpoint loaded in CPU')
        self.model = model
        self.model.eval()
        vocab_manager = VocabularyManager()
        self.vocab_size = vocab_manager.vocab_size
        self.penalty = penalty
        self.topk = topk
        self.top_p = top_p
        self.temperature = temperature
        self.tokenizer = Tokenizer()
        def predict(inputs):
            with torch.no_grad():
                inputs = torch.tensor(inputs)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                output = self.model(inputs)
                output = output[:,:,-26:-1] # remove non-AA token logits
                return output
                
        self.predict = predict

        def predict_with_stop(inputs):
            with torch.no_grad():
                inputs = torch.tensor(inputs)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()            
                output = self.model(inputs)
                stop_token = output[:, :, 4] # the stop token logits
                output = output[:,:,-26:-1] # remove non-AA token logits
                return output, stop_token
        self.predict_with_stop = predict_with_stop
        
        def predict_stop_mine(inputs):
            with torch.no_grad():
                inputs = torch.tensor(inputs)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                output = self.model(inputs) # remove non-AA token logits but keep stop at last position 
                stop_token = output[:,:,4:5] # get stop token logits
                output = output[:,:,-26:-1] # remove non-AA token logits but keep stop at last position
                return torch.concatenate((output, stop_token), axis = -1)
        self.predict_stop_mine = predict_stop_mine  


    def techer_forcing_generation_new(self, input_dataset_path='/home/saitto/ProGen/my_data/test.p'):
        # input: path to the dataset
        # output: list of true sequences, list of generated sequences (top-k implementation),  
        #         list of list of aa prob distribution.
        with open(input_dataset_path, 'rb') as f:
            data = pickle.load(f)
        names = list(data.keys())
        # for testing:
        #names = names[:64]
        keywords = [data[name]["kw"] for name in names]
        begAAindex = [len(keywords[n]) for n in range(len(keywords))]
        sequences = [data[name]["seq"] for name in names]
        inputs = []
        for n in range (len(sequences)):
            keywords[n] = [int(kw) for kw in keywords[n]]
            inputs.append(keywords[n] + [self.tokenizer.aa_to_ctrl_idx[aa] for aa in sequences[n]])
        padIndex = [len(input) for input in inputs]
        maximum = max([len(i) for i in inputs])
        input_tensor = torch.zeros(len(inputs), maximum, dtype=torch.long)
        for n, input in enumerate(inputs):
            input_tensor[n, :len(input)] = torch.tensor(input, dtype=torch.long)
        input_tensor = input_tensor.cuda()
        batch_size = 64  # Adjust based on your GPU memory
        n_batches = (input_tensor.shape[0] + batch_size - 1) // batch_size  # Compute the number of batches needed
        full_output = torch.zeros(len(input_tensor), maximum, 25, dtype=torch.float32).cuda()  # Initialize an empty tensor to store the predictions
        with torch.no_grad():  # Ensure gradients aren't computed for efficiency
            for i in range(n_batches):
                if i == n_batches - 1:
                    batch_end = input_tensor.shape[0]
                    batch_start = i * batch_size
                else:
                    batch_start, batch_end = i * batch_size, (i + 1) * batch_size
                
                batch = input_tensor[batch_start:batch_end, :]
                output = self.model(batch)[:,:,-26:-1]
                full_output[batch_start:batch_end, :, :] = output
        # tensore full_output: num sequenze, valore max len lunghezza sequenza, 25 (output delle probabilità)
        full_output = full_output.cpu()
        generated_sequences = []
        probs_aa_distributions = []
        for sample_i in range(full_output.shape[0]):
            if begAAindex[sample_i] == 0:
                y_pred = full_output[sample_i][:-1]
            else: # if there is a keyword
                y_pred = full_output[sample_i][begAAindex[sample_i]-1:padIndex[sample_i]-1]
            y_pred = F.softmax(y_pred, dim=1)
            
            y_pred = y_pred.numpy()
            pruned_array = np.argsort(y_pred, axis = 1)[:,::-1] # sort the probabilities in descending order

            if self.topk == 1: #we generate the sequence now given the highest probability tokens from the pruned array
                final_array = pruned_array[:, 0]
            else: #we generate the sequence sampling from the top k tokens
                final_array = np.zeros(pruned_array.shape[0])
                for n in range(pruned_array.shape[0]): #for every token to be generated
                    #I will normalize the probabilities of the top k tokens and then sample from them
                    normalizer = np.sum(y_pred[n, [pruned_array[n, :self.topk]]].flatten())
                    p = y_pred[n, [pruned_array[n, :self.topk]]].flatten()/normalizer
                    final_array[n] = np.random.choice(pruned_array[n, :self.topk], p = p)
            #We also need to convert back the control code indices to the aminoacids
            generated_sequence = "".join([self.tokenizer.ctrl_idx_to_aa[(self.vocab_size-26) + x] for i, x in enumerate (final_array)])
    
            prompt_probs_lst = []
            for i in range(y_pred.shape[0]):
                prompt_probs_lst.append(list(y_pred[i,:]))
            generated_sequences.append(generated_sequence)
            probs_aa_distributions.append(prompt_probs_lst)
        
        return sequences, generated_sequences, probs_aa_distributions
        
    def teacher_forcing_generation(self, input_sequence, tax_lineage):
        res = ""
        tokens_prob = []
        key_len = len(tax_lineage) # len(kw_lineage+tax_lineage)
        print(input_sequence)
        for i in range(1, len(input_sequence)):
            iteration_input_prefix = input_sequence[:i]
            seed_seq = [self.tokenizer.aa_to_ctrl_idx[ii] for ii in iteration_input_prefix]
            generate_num = key_len + len(seed_seq) + 1 # how many tokens to generate, here only one
            seq_length = min(generate_num, 511)
            text = tax_lineage + seed_seq # tax_lineage + kw_lineage + seed_seq
            padded_text = text + [0] * (generate_num - len(text))
            tokens_generated = np.tile(padded_text, (1,1))
            for token in range(len(text)-1, generate_num-1):
                prompt_logits = self.predict(tokens_generated[:, :seq_length]).squeeze()
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
            prompt_logits = self.predict(tokens_generated[:, :seq_length]).squeeze()
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
        return tokens_generated, tokens_prob, n

    def generation_complete_sequence_new(self, input_sequence: str, keywords: list, starting_aa: int, max_len=512) -> tuple[str, list]:
        '''
        input_sequence: the input sequence to be used for the generation
        keywords: the keywords of the input sequence
        starting_aa: the position in the input sequence from which the generation will start
        top_p: the probability threshold to be used for the generation
        max_len: the maximum length of the sequence to be generated if the input sequence is shorter than 511
        Returns the generated sequence
        Returns a list of sequences generated when the stop token is predicted
        Returns the probability distributions of the tokens generated as a list of lists
        '''
        input_indices = [self.tokenizer.aa_to_ctrl_idx[j] for j in input_sequence]
        if len(input_indices)  < max_len:
            #the length of the input array is: len(keywords) + max_len
            total_array = np.array(keywords + input_indices + [0]*(max_len - len(input_indices))).reshape(1, -1)
        else:
            total_array = np.array(keywords + input_indices).reshape(1, -1)

        tokens_prob = []
        res_stopped = []
        
        #due to a bug in torch, we need to run the model with a dummy of max_len + len(keywords) to avoid a bug
        dummy = np.zeros((1, max_len + len(keywords)), dtype=int)
        _ = self.predict_stop_mine(dummy).squeeze().cpu().detach()

        #for every amino acid in the input sequence starting from the starting position and ending with the last token to be generated
        for current_position in range(len(keywords)+ starting_aa, total_array.shape[1]):
            if current_position == 1: #the output is already one dimensional
                prompt_logits = self.predict_stop_mine(total_array[:,:current_position]).squeeze().cpu().detach().numpy()
            elif current_position <= 511:
                prompt_logits = self.predict_stop_mine(total_array[:,:current_position]).squeeze().cpu().detach().numpy()[-1]       
            else: #if the input sequence is longer than 511, we will use only the last 511 - len(keywords) aminoacids
                start = current_position - 511 + len(keywords)
                keyword_indices = [x for x in range(len(keywords))]
                sequence_indices = [x for x in range(start, current_position)]
                prompt_logits = self.predict_stop_mine(total_array[:,keyword_indices + sequence_indices]).squeeze().cpu().detach().numpy()[-1]
            
            prompt_logits = prompt_logits/(self.temperature if self.temperature>0 else 1.)  

            if self.penalty>0:
                input_positions = list(total_array[0][:] - (self.vocab_size-26))
                # look at the previous 4 positions (if available)
                if current_position >= 4 + len(keywords):
                    input_to_check = set(input_positions[current_position-4:current_position])
                else:
                    input_to_check = set(input_positions[len(keywords):current_position])
                for token in input_to_check:
                    prompt_logits[token] /= self.penalty
                    
            softmax = lambda logits: np.exp(logits) / np.sum(np.exp(logits))
            # probability of the stop token
            prob_stop = softmax(prompt_logits)[-1]
            # probabilities of the AA
            prompt_probs_no_stop = softmax(prompt_logits[:-1])
            pruned_array = np.argsort(prompt_probs_no_stop)[::-1]

            if prob_stop >= 0.50:
                tokens_generated_stopped = list(total_array[0][len(keywords):current_position])
                tokens_generated_stopped = ''.join([self.tokenizer.ctrl_idx_to_aa[c] for c in tokens_generated_stopped])
                res_stopped.append(tokens_generated_stopped)
            
            #top_p is used to sample from the tokesns where the cumulative probability is less than or equal to p
            if self.top_p==1:
                total_array[0][current_position] = pruned_array[0]  + (self.vocab_size-26)
            else:
                sorted_probs, sorted_indices = torch.sort(torch.tensor(prompt_probs_no_stop), descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                valid_indices = sorted_indices[cum_probs <= self.top_p]
                if valid_indices.size(0) == 0:
                    valid_indices = sorted_indices[:1]
                total_array[0][current_position] = valid_indices[torch.randint(0, valid_indices.size(0), (1,))].item() + (self.vocab_size-26)
            tokens_prob.append(prompt_probs_no_stop.tolist())
        final_array = total_array[0][len(keywords):].tolist()
        generated_sequence = "".join([self.tokenizer.ctrl_idx_to_aa[x] for x in final_array])
        
        return generated_sequence, res_stopped, tokens_prob
    
    def generation_complete_sequence(self, input_sequence, after_n, tax_lineage):
        res = ""
        res_stopped = []
        # tokens_prob = []
        key_len = len(tax_lineage) # len(kw_lineage+tax_lineage)
        i = after_n # if we have a sequence of amminoacids in input, we can add some in input as seed sequence
        iteration_input_prefix = input_sequence[:i]
        seed_seq = [self.tokenizer.aa_to_ctrl_idx[ii] for ii in iteration_input_prefix]
        generate_num = key_len + (len(input_sequence)) # how many tokens to generate
        if generate_num < 511:
            generate_num = 511
        seq_length = min(generate_num, 511)
        text = tax_lineage + seed_seq # tax_lineage + kw_lineage + seed_seq
        padded_text = text + [0] * (generate_num - len(text))
        tokens_generated = np.tile(padded_text, (1,1))
        for token in range(len(text)-1, generate_num-1):
            
            # prediction
            prompt_logits, stop_token = self.predict_with_stop(tokens_generated[:, :seq_length])
            prompt_logits = prompt_logits.squeeze()  / (self.temperature if self.temperature>0 else 1.)
            stop_token = stop_token.squeeze()  / (self.temperature if self.temperature>0 else 1.)
            
            _token = token if token < seq_length else -1
            prompt_logits = prompt_logits.cpu().detach().numpy()
            stop_token = stop_token.cpu().detach().numpy()
            
            # penalty
            if (self.penalty>0) and (token >= key_len + 3):
                penalized_so_far = set()
                for _ in range(token-3,token+1):
                    generated_token = tokens_generated[0][_] - (self.vocab_size-26) # added
                    if generated_token in penalized_so_far:
                        continue
                    penalized_so_far.add(generated_token)
                    prompt_logits[_token][generated_token] /= self.penalty  
    
            # compute probabilities from logits
            prompt_probs = np.exp(prompt_logits[_token])
            prompt_probs = prompt_probs / sum(prompt_probs)
    
            # ESTRARRE TOKEN 1: the stop token, softmax con le probabilità degli amminoacidi
            logits_and_stop = np.concatenate((prompt_logits[_token], [stop_token[_token]]))
            logits_and_stop_prob = np.exp(logits_and_stop)
            logits_and_stop_prob = logits_and_stop_prob / sum(logits_and_stop_prob)
    
            if logits_and_stop_prob[-1] >= 0.50:
                tokens_generated_stopped = tokens_generated[0][len(seed_seq) + key_len:_token + 1]
                tokens_generated_stopped = ''.join([self.tokenizer.ctrl_idx_to_aa[c] for c in tokens_generated_stopped])
                res_stopped.append(tokens_generated_stopped)
            
            pruned_list = np.argsort(prompt_probs)[::-1]
            # tokens_prob.append([prompt_probs.tolist()])
    
            if self.top_p==1:
                idx = pruned_list[0]
            else:
                # Sort the probabilities
                sorted_probs, sorted_indices = torch.sort(torch.tensor(prompt_probs), descending=True)
                # Calculate cumulative probs
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                # Get the set of tokens whose cumulative probability is less than or equal to p
                valid_indices = sorted_indices[cum_probs <= self.top_p]
                # If no token's cumulative probability is less than the threshold, just select the top token
                if valid_indices.size(0) == 0:
                    valid_indices = sorted_indices[:1]
                # Sample from the valid indices
                idx = valid_indices[torch.randint(0, valid_indices.size(0), (1,))].item()
            # assign the token for generation
            idx += (self.vocab_size-26) # added to convert 0 AA to original ctrl idx
            tokens_generated[0][token+1] = idx
            
        tokens_generated = tokens_generated[0][len(seed_seq) + key_len:]
        tokens_generated = ''.join([self.tokenizer.ctrl_idx_to_aa[c] for c in tokens_generated])
        return tokens_generated, res_stopped
