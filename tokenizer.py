'''
Tokenizer for the ProGen model
'''
import os
import pickle
MAPPING_FOLDER = 'mapping_files/'
KW_FILE = 'kw_mapping.p'
AA_FILE = 'aa_mapping.p'

class Tokenizer:
    def __init__(self):
        with open(MAPPING_FOLDER + KW_FILE, 'rb') as handle:
            self.kw_to_ctrl_idx = pickle.load(handle)
        with open(MAPPING_FOLDER + AA_FILE, 'rb') as handle:
            self.aa_to_ctrl_idx = pickle.load(handle)

        def flipdict(my_map):
            return {v: k for k, v in my_map.items()}
        self.ctrl_idx_to_aa = flipdict(self.aa_to_ctrl_idx)
        self.ctrl_idx_to_kw = flipdict(self.kw_to_ctrl_idx)