import torch
from torch.utils.data import Dataset
from transformProtein import transformProtein
import numpy as np
import pickle

class ProteinDataset(Dataset):

    def __init__(self, pklpath, firstAAidx, transformFull=None, evalTransform=None):
        with open(pklpath, 'rb') as handle:
            self.data_chunk = pickle.load(handle)
        self.uids = list(self.data_chunk.keys())
        self.transformFull = transformFull
        self.evalTransform = evalTransform
        self.firstAAidx = firstAAidx
        
        self.trainmode=False
        if self.evalTransform==None: self.trainmode=True

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        if self.trainmode:
            transformObj = self.transformFull
        else:
            transformObj = self.evalTransform

        sample_arr, existence, padIndex = transformObj.transformSample(self.data_chunk[self.uids[idx]])
        sample_arr = np.array(sample_arr)
        inputs = sample_arr[:-1]
        outputs = sample_arr[1:]
        
        begAAindex = np.argwhere(inputs>=self.firstAAidx)[0][0]
        
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs, existence, padIndex, begAAindex
    
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    # Define the path to one pickle files
    pklpath = 'data/train_test_pkl/train0.p'
    
    # instance of the transformProtein class
    transform_obj = transformProtein(maxSampleLength = 511+1, maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.0)
    
    # load the vocabulary from file
    vocab = open('mapping_files/vocab.txt').readlines()
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    # length of the vocabulary
    vocab_size = len(vocab)
    firstAAidx = vocab_size - 26
    
    # Create an instance of the ProteinDataset class    
    dataset = ProteinDataset(pklpath, firstAAidx = firstAAidx, transformFull = transform_obj)
    
    
    dataloader = DataLoader(dataset, shuffle = True, batch_size = 1,
                                        num_workers = 0, pin_memory = False) 
                
    for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
        # Print the details of the sample
        print("sample:", sample)
        print("labels:", labels)
        print("Existence:", existence)
        print("Padding Index:", padIndex)
        print("Beginning Amino Acid Index:", begAAindex)
        break

