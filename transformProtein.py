import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tokenizer import Tokenizer
from model_manager import VocabularyManager

class transformProtein:
    def __init__(self, mapfold = 'mapping_files/', maxSampleLength = 512,
                 verbose = False, maxTaxaPerSample = 3, 
                 maxKwPerSample = 5, dropRate = 0.0, seqonly = False, noflipseq = False):

        self.maxSampleLength = maxSampleLength
        self.verbose = verbose
        self.maxTaxaPerSample = maxTaxaPerSample
        self.maxKwPerSample = maxKwPerSample
        self.dropRate = dropRate
        self.seqonly = seqonly
        self.noflipseq = noflipseq
        self.tokenizer = Tokenizer()
        vocab_manager = VocabularyManager()
        self.oneEncoderLength = vocab_manager.vocab_size -1
    
    def transformSeq(self, seq, prob = 0.0):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>(1-prob):
            seq = seq[::-1]
        return seq

    def transformKwSet(self, kws, drop = 0.1):
        """
        Filter kws, dropout, and replace with lineage (including term at end)
        """
        # kws = [i for i in kws if i in self.tokenizer.kw_to_ctrl_idx]
        # np.random.shuffle(kws)
        kws = kws[:self.maxKwPerSample]
        kws = [i for i in kws if np.random.random()>drop]
        return kws

    def transformTaxaSet(self, taxa, drop = 0.1):
        """
        Filter taxa, dropout, and replace with lineage (including term at end)
        """
        taxa = [i for i in taxa if i in self.tokenizer.taxa_to_ctrl_idx]
        np.random.shuffle(taxa)
        taxa = taxa[:self.maxTaxaPerSample]
        
        taxa = [self.tokenizer.taxa_to_lineage[i] for i in taxa if np.random.random()>drop]
        
        return taxa

    def transformSample(self, proteinDict):
        """
        Function to transform/augment a sample.
        Padding with all zeros
        Returns an encoded sample (taxa's,kw's,sequence) and the existence level to multiply weights
        """
        stop_token = 4
        existence = 1
        if (not self.seqonly):
            kws = self.transformKwSet(proteinDict['kw'], drop = self.dropRate)
            print(kws)
            if proteinDict['ex'] in [4, 5]:
                existence += 1
            if proteinDict['rev']:
                existence += 1
        seq = self.transformSeq(proteinDict['seq'])

        if self.seqonly:
            encodedSample = []
            seq_idx = 0
            while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                encodedSample.append(self.tokenizer.aa_to_ctrl_idx[seq[seq_idx]])
                seq_idx += 1
            if len(encodedSample)<self.maxSampleLength:
                encodedSample.append(stop_token)
            while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                encodedSample.append(self.oneEncoderLength)
            if self.verbose:
                print(seq)
                print(encodedSample)
            return encodedSample
            
        encodedSample = []
        for kw_line in kws:
            encodedSample.extend(kw_line)
        seq_idx = 0
        while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
            encodedSample.append(self.tokenizer.aa_to_ctrl_idx[seq[seq_idx]])
            seq_idx += 1
        if len(encodedSample)<self.maxSampleLength:
            encodedSample.append(stop_token)
        thePadIndex = len(encodedSample)
        while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
            encodedSample.append(self.oneEncoderLength)
                
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
    with open('data_halogenase/chunks/train'+str(chunknum)+'.p','rb') as handle:
        train_chunk = pickle.load(handle)
    #uid = 'UPI000000BF1A'
    #uid = random.sample(train_chunk.keys(),1)[0]
    obj = transformProtein(verbose=True, dropRate = 0.1, maxTaxaPerSample = 3, maxKwPerSample = 5, seqonly = False)
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
