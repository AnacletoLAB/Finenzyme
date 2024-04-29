'''
Tokenizer for the ProGen model
'''
import os
import pickle

class Tokenizer:
    def __init__(self):
        with open('mapping_files/aa_to_ctrl_idx.p','rb') as handle:
            self.aa_to_ctrl_idx = pickle.load(handle)

    
        def flipdict(my_map):
            return {v: k for k, v in my_map.items()}
            
        self.ctrl_idx_to_aa = flipdict(self.aa_to_ctrl_idx)

        