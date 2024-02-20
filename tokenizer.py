'''
Tokenizer for the ProGen model
'''
import os
import pickle

class Tokenizer:
    def __init__(self):
        with open(os.path.join('mapping_files/','taxa_to_lineage.p'),'rb') as handle:
            self.taxa_to_lineage = pickle.load(handle)
        with open('mapping_files/taxa_to_ctrl_idx.p','rb') as handle:
            self.taxa_to_ctrl_idx = pickle.load(handle)
        with open('mapping_files/kw_to_ctrl_idx.p','rb') as handle:
            self.kw_to_ctrl_idx = pickle.load(handle)
        with open('mapping_files/aa_to_ctrl_idx.p','rb') as handle:
            self.aa_to_ctrl_idx = pickle.load(handle)
        with open('mapping_files/kw_to_name.p2','rb') as handle:
            self.kw_to_name = pickle.load(handle)
        with open('mapping_files/probs_to_aa.p', 'rb') as handle:
            self.aa_to_probs_index = pickle.load(handle)
        # with open('mapping_files/taxid_to_name.p2','rb') as handle:
        # taxid_to_name = pickle.load(handle)
    
        def flipdict(my_map):
            return {v: k for k, v in my_map.items()}
            
        self.ctrl_idx_to_aa = flipdict(self.aa_to_ctrl_idx)
        self.ctrl_idx_to_kw = flipdict(self.kw_to_ctrl_idx)
        self.ctrl_idx_to_taxa = flipdict(self.taxa_to_ctrl_idx)
        self.probs_index_to_aa = flipdict(self.aa_to_probs_index)
        