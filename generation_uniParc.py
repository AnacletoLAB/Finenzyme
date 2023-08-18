from __future__ import print_function
from __future__ import division
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
GPU = torch.cuda.is_available()
print(GPU)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
# GPU = False
# METAPRECISION = False

import os
import sys
import torch
import tqdm
import pdb
import numpy as np
import platform
import hashlib
import pytorch_transformer
import re
import argparse
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from transformProtein import transformProtein
from ProteinDataset_uid import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import matplotlib.pyplot as plt

def check_gpu_processes():
    gpu_processes = os.popen("nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits").read().strip().split('\n')
    return len(gpu_processes) == 0
def wait_for_gpu_idle():
    while not check_gpu_processes():
        print("GPU is busy. Waiting...")
        time.sleep(240)  # Adjust the interval as needed
    print("GPU is idle. Proceeding with the script.")

# Call the function to wait for GPU to be idle
wait_for_gpu_idle()

torch.cuda.empty_cache()
load_model_path = 'ckpt/' # just the folder itself

seq_length = 511
embedding_dim = 1280
num_layers = 36
vocab_loc = 'mapping_files/vocab.txt'

use_py3 = platform.python_version()[0] == '3'
vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

class TiedEmbeddingSoftmax(torch.nn.Module):

    def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
        super(TiedEmbeddingSoftmax, self).__init__()
        self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
        self.b = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs, embed=True):
        with autocast(enabled=METAPRECISION):
            if embed:
                return torch.nn.functional.embedding(inputs, self.w)
            else:
                return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
    def __init__(self):
        super(CTRLmodel,self).__init__()
        self.tied_embedding_softmax = TiedEmbeddingSoftmax()
        self.encoder = pytorch_transformer.Encoder()

    def forward(self, inputs):
        with autocast(enabled=METAPRECISION):
            x = self.tied_embedding_softmax(inputs, embed=True)
            x = self.encoder(x)
            x = self.tied_embedding_softmax(x, embed=False)
        return x

    def loadCheckpoint(self, model_path, num_layers):
        if os.path.exists(model_path):
            print('Found PyTorch checkpoint at ', model_path)
            print('Loading instead of converting from TensorFlow')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.tied_embedding_softmax.load_state_dict({
                'w': checkpoint.pop('tied_embedding_softmax.w', None),
                'b': checkpoint.pop('tied_embedding_softmax.b', None)
            })
            self.encoder.load_state_dict({key.replace("encoder.", ""): value for key, value in checkpoint.items()})
        else:
            print('Could not find PyTorch checkpoint')
            sys.exit()

model = CTRLmodel()
print('model initialized')

curr_model_path = load_model_path+'pretrain_progen_full.pth'
reader = model.loadCheckpoint(model_path=curr_model_path, num_layers = num_layers)
print('previous checkpoint loaded')
            
if METAPRECISION:
    model = model.to(torch.float16)
if GPU:
    model = model.cuda()
    print('previous checkpoint loaded in GPU')

optimizer = torch.optim.Adam(model.parameters()) #lr, betas

model.eval()

with open(os.path.join('mapping_files/','taxa_to_lineage.p'),'rb') as handle:
    taxa_to_lineage = pickle.load(handle)
with open('mapping_files/taxa_to_ctrl_idx.p','rb') as handle:
    taxa_to_ctrl_idx = pickle.load(handle)
with open('mapping_files/kw_to_ctrl_idx.p','rb') as handle:
    kw_to_ctrl_idx = pickle.load(handle)
with open('mapping_files/aa_to_ctrl_idx.p','rb') as handle:
    aa_to_ctrl_idx = pickle.load(handle)
with open('mapping_files/kw_to_name.p2','rb') as handle:
    kw_to_name = pickle.load(handle)
#with open('mapping_files/taxid_to_name.p2','rb') as handle:
#    taxid_to_name = pickle.load(handle)
# MANCANO:
# kw_to_lineage.p
# kw_to_parents.p
# taxa_to_parents.p
    
def flipdict(my_map):
    return {v: k for k, v in my_map.items()}
ctrl_idx_to_aa = flipdict(aa_to_ctrl_idx)
ctrl_idx_to_kw = flipdict(kw_to_ctrl_idx)
ctrl_idx_to_taxa = flipdict(taxa_to_ctrl_idx)

def predict_fn(inputs):
    with torch.no_grad():
        inputs = torch.tensor(inputs)
        if GPU:
            inputs = inputs.cuda()            
        output = model(inputs)
        output = output[:,:,-26:-1] # remove non-AA token logits
        return output


# In[8]:


import pickle
import os
# paths to the saved .p files
random_selection_file = os.path.join("data_random_selection_uniParc", "random_selection_2000.p")
#data_file = os.path.join("data", "filtered_data_" + query + ".p")

# Reload random_selection from the .p file
random_selection = False
with open(random_selection_file, "rb") as file:
    random_selection = pickle.load(file)

# Reload filtered_data from the .p file
#filtered_data = False
#with open(data_file, "rb") as file:
#    filtered_data = pickle.load(file)

# Print or use the reloaded datasets as needed
print("Random Selection:")
for entry in random_selection:
    print(entry["sequence"]["value"])
    print(len(random_selection))
    print(entry)
    break

#print("---")

#print("Filtered Data:")
#for entry in filtered_data:
#    print(entry["sequence"])'''


# In[6]:


"""for entry in random_selection:
    taxid = int(entry['metadata']['source_organism']['taxId']) # taxonomy id from NCBI
    print(taxid)
    try:
        tax_lineage = taxa_to_lineage[taxid] # make lineage in ncbi ids
        print(tax_lineage)
        tax_lineage = [taxa_to_ctrl_idx[ite] for ite in tax_lineage] # now translated as ctrl code indices
        print(tax_lineage)
    except:
        print("Tax lineage error")
    # break"""


# <p>To evaluate the generated sequence using teacher forcing:
# 
# 1. Generate the Sequence: The transformer model with teacher forcing generates the complete sequence. 
# At each time step, we provide the true input sequence (+ 1 versus previous step), as input to the model.
# 
# 2. Calculate Metrics: Accuracy, soft accuracy (based on BLOSUM matrix), perplexity. 
# 
# </p>

# In[10]:


def teacher_forcing_generation(input_sequence, tax_lineage, penalty, topk):
    key_len = 0
    res = ""
    tokens_prob = []
    true_tokens_index_in_prob = []
    key_len = len(tax_lineage) # len(kw_lineage+tax_lineage)
    for i in range(1, len(input_sequence)):
        iteration_input_prefix = input_sequence[:i]
        seed_seq = [aa_to_ctrl_idx[ii] for ii in iteration_input_prefix]
        generate_num = key_len + len(seed_seq) + 1 # how many tokens to generate, here only one
        seq_length = min(generate_num, 511)
        text = tax_lineage + seed_seq # tax_lineage + kw_lineage + seed_seq
        padded_text = text + [0] * (generate_num - len(text))
        tokens_generated = np.tile(padded_text, (1,1))
        for token in range(len(text)-1, generate_num-1):
            prompt_logits = predict_fn(tokens_generated[:, :seq_length]).squeeze()
            _token = token if token < seq_length else -1
            prompt_logits = prompt_logits.cpu().detach().numpy()
            #print(tokens_generated[:, :seq_length])
            #print(prompt_logits)
            if penalty>0:
                penalized_so_far = set()
                # variable token_flag for first amminoacids (to count them if they are less that 4)
                if token >= key_len + 3:
                    token_flag = 3  
                elif token - key_len - 3 <= 0:
                    token_flag = 0
                else:
                    token_flag = token
                for _ in range(token-token_flag,token+1):
                    generated_token = tokens_generated[0][_] - (vocab_size-26) # added
                    if generated_token in penalized_so_far:
                        continue
                    penalized_so_far.add(generated_token)
                    prompt_logits[_token][generated_token] /= penalty  
            # compute probabilities from logits
            prompt_probs = np.exp(prompt_logits[_token])
            prompt_probs = prompt_probs / sum(prompt_probs)
            pruned_list = np.argsort(prompt_probs)[::-1]
            codice_aa_della_true_sequence = seed_seq[_token - key_len]
            idx_true = codice_aa_della_true_sequence - vocab_size + 26
            true_tokens_index_in_prob.append([idx_true])
            tokens_prob.append([prompt_probs.tolist()])

            if topk==1:
                idx = pruned_list[0]
            else:
                pruned_list = pruned_list[:topk]
                chosen_idx = torch.distributions.categorical.Categorical(logits=torch.tensor(np.expand_dims(prompt_logits[_token][pruned_list],0))).sample().numpy()[0]
                idx = pruned_list[chosen_idx]
            # assign the token for generation
            idx += (vocab_size-26) # added to convert 0 AA to original ctrl idx
            tokens_generated[0][token+1] = idx
        tokens_generated_so_far = tokens_generated[0].squeeze()[:token+2]
        tokens_generated_so_far = tokens_generated_so_far[(tokens_generated_so_far>=(vocab_size-26)) & (tokens_generated_so_far<(vocab_size-1))]
        tokens_generated_so_far = ''.join([ctrl_idx_to_aa[c] for c in tokens_generated_so_far])
        query = tokens_generated_so_far[len(seed_seq):]
        res += query
    return res, tokens_prob, true_tokens_index_in_prob


# In[11]:


import pickle
# print("on data: ", query)
penalty = 1.2
topk = 3
predicted = []
true_tokens_index_in_probs_all = []
tokens_probs_all = []
true_value = []
for entry in random_selection:
    input_seq = entry['sequence']['value']
    taxid = None # taxonomy id from NCBI DA IMPLEMENTARE
    """try:
        tax_lineage = taxa_to_lineage[taxid] # make lineage in ncbi ids
        tax_lineage = [taxa_to_ctrl_idx[ite] for ite in tax_lineage] # now translated as ctrl code indices
    except:
        print("ALERT: error in taxonmy conversions for entry: ", taxid)
        try:
            tax_lineage = taxa_to_ctrl_idx[taxid]
        except:
            print("ALERT: taxonomy id error for entry: ", taxid)
            try:
                tax_lineage = [taxa_to_ctrl_idx[ite] for ite in tax_lineage[:-1]]
                print("ALERT: taxonomy id error for entry: ", taxid, ", but lineage found.")
            except:
                print("ALERT: taxonomy id error for entry: ", taxid, ", and error of lineage.")
                tax_lineage = []"""
    # FOR TESTIG the code:
    # input_seq = input_seq[0:4]
    
    # if no tax keys:
    tax_lineage = []
    
    try:
        res, tokens_prob, true_tokens_index_in_prob = teacher_forcing_generation(input_seq, tax_lineage,
                                                                                 penalty, topk)
    except:
        print("ALERT: skipped for error the entry: ", entry)
        continue
    # print("Input: ", input_seq)
    # print("Res: ", res)
    true_value.append(input_seq)
    true_tokens_index_in_probs_all.append(true_tokens_index_in_prob)
    tokens_probs_all.append(tokens_prob)
    predicted.append(input_seq[0] + res)

    # FOR TESTIG:
    #break
    
    


# In[12]:


# Create a directory named "ID_test_data" in the current working directory if it doesn't exist
data_dir = "data_random_selection_uniParc/2000_sequences_p_1.2_topk_3"
os.makedirs(data_dir, exist_ok=True)

# TODO, salvare tokens_prob e true_tokens_index_in_prob
tokens_probs_data_file = os.path.join(data_dir, "tokens_probs_data.p")
with open(tokens_probs_data_file, "wb") as file:
    pickle.dump(tokens_probs_all, file)
    
true_tokens_index_data_file = os.path.join(data_dir, "true_tokens_index_data.p")
with open(true_tokens_index_data_file, "wb") as file:
    pickle.dump(true_tokens_index_in_probs_all, file)

# Save ID_test_data as a .p file
predicted_data_file = os.path.join(data_dir, "predicted_data.p")
with open(predicted_data_file, "wb") as file:
    pickle.dump(predicted, file)
    
# Save ID_test_data as a .p file
true_data_file = os.path.join(data_dir, "true_data.p")
with open(true_data_file, "wb") as file:
    pickle.dump(true_value, file)
torch.cuda.empty_cache()
