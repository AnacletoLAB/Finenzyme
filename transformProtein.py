import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random

class transformProtein:
    def __init__(self, mapfold = 'mapping_files/', maxSampleLength = 512, selectSwiss = 1.0,
                 selectTrembl = 1.0, verbose = False, maxTaxaPerSample = 3, 
                 maxKwPerSample = 5, dropRate = 0.0, seqonly = False, noflipseq = False):

        self.maxSampleLength = maxSampleLength
        self.selectSwiss = selectSwiss
        self.selectTrembl = selectTrembl
        self.verbose = verbose
        self.maxTaxaPerSample = maxTaxaPerSample
        self.maxKwPerSample = maxKwPerSample
        self.dropRate = dropRate
        self.seqonly = seqonly
        self.noflipseq = noflipseq

        if self.seqonly:
            with open(os.path.join(mapfold,'aa_to_ctrl_idx_seqonly.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            print('Sequence only - no CTRL codes')
            self.oneEncoderLength = max(self.aa_to_ctrl.values())+1
        else:
            with open(os.path.join(mapfold,'kw_to_ctrl_idx.p'),'rb') as handle:
                self.kw_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_ctrl_idx.p'),'rb') as handle:
                self.taxa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_parents.p'),'rb') as handle: # TODO: remove
                self.taxa_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_parents.p'),'rb') as handle: # TODO: remove
                self.kw_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_lineage.p'),'rb') as handle:
                self.kw_to_lineage = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_lineage.p'),'rb') as handle:
                self.taxa_to_lineage = pickle.load(handle)
    
            self.oneEncoderLength = max(max(self.kw_to_ctrl.values()),max(self.taxa_to_ctrl.values()),max(self.aa_to_ctrl.values())) + 1
            print('Using one unified encoder to represent protein sample with length', self.oneEncoderLength)
    
    def transformSeq(self, seq):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>=0.8:
            seq = seq[::-1]
        return seq

    def transformKwSet(self, kws, drop = 0.1):
        """
        Filter kws, dropout, and replace with lineage (including term at end)
        """
        # kws = [i for i in kws if i in self.kw_to_ctrl]
        np.random.shuffle(kws)
        kws = kws[:self.maxKwPerSample]
        
        if np.random.random()<=drop:
            kws = []
      
        return kws

    def transformTaxaSet(self, taxa, drop = 0.1):
        """
        Filter taxa, dropout, and replace with lineage (including term at end)
        """
        taxa = [i for i in taxa if i in self.taxa_to_ctrl]
        np.random.shuffle(taxa)
        taxa = taxa[:self.maxTaxaPerSample]
        
        taxa = [self.taxa_to_lineage[i] for i in taxa if np.random.random()>drop]
        
        return taxa

    def transformSample(self, proteinDict, justidx = True):
        """
        Function to transform/augment a sample.
        If it's not in swiss or trembl, existence set to 3. Or if it's found in swiss/trembl and you sample an other taxa.
        Padding with all zeros
        Returns an encoded sample (taxa's,kw's,sequence) and the existence level
        """
        existence = 3
        if (not self.seqonly):
            existence = 3
            kws = self.transformKwSet(proteinDict['kw'], drop = self.dropRate)
        elif (not self.seqonly):
            kws = {}
        seq = self.transformSeq(proteinDict['seq'])

        if self.seqonly:
            encodedSample = []
            seq_idx = 0
            while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                seq_idx += 1
            stop_token = 1
            if len(encodedSample)<self.maxSampleLength:
                encodedSample.append(stop_token)
            while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                encodedSample.append(self.oneEncoderLength)
            if self.verbose:
                print(seq)
                print(encodedSample)
            return encodedSample
            
        encodedSample = []
        if self.oneEncoderLength:
            if justidx:
                for kw_line in kws:
                    encodedSample.extend(kws)
                seq_idx = 0
                while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                    encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                    seq_idx += 1
                stop_token = 1
                if len(encodedSample)<self.maxSampleLength:
                    encodedSample.append(stop_token)
                thePadIndex = len(encodedSample)
                while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                    encodedSample.append(self.oneEncoderLength)
            else: 
                os.exit('error fatal')
                
        if self.verbose:
            print('Raw Data')
            for k in proteinDict:
                print('--------',k,'--------')
                print(proteinDict[k])
            print('Transformed Sample -------')
            print('Seq',seq)
            print('Existence', existence)
            print('KWs',kws)
            print('encodedSample',encodedSample)
            print('thePadIndex', thePadIndex)
            
        return encodedSample, existence, thePadIndex

if __name__ == "__main__":
    chunknum = 0
    with open('data/train_test_pkl/train'+str(chunknum)+'.p','rb') as handle:
        train_chunk = pickle.load(handle)
    #uid = 'UPI000000BF1A'
    #uid = random.sample(train_chunk.keys(),1)[0]
    obj = transformProtein(verbose=True, dropRate = 0.1, maxTaxaPerSample = 3, maxKwPerSample = 5, 
                           selectSwiss = 1.0, selectTrembl = 1.0, seqonly = False)
    i = 0
    for key in train_chunk:
        encodedSample, existence, thePadIndex = obj.transformSample(train_chunk[key])
        print('max sample len', obj.maxSampleLength)
        print('encodedSample: ', encodedSample)
        print('existence: ', existence)
        print('thePadIndex', thePadIndex)
        print('encodedSample + pad index: ', encodedSample[:thePadIndex])
        print('encodedSample - pad index: ', encodedSample[thePadIndex:])
        i += 1
        if i == 2:
            break
