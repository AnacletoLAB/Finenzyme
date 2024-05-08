import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tokenizer import Tokenizer
from model_manager import VocabularyManager

MAPPING_FOLDER = 'mapping_files/'
DATA_PATH = 'scop_data/'
DATA_FILE = 'training_scop.p'

class transformProtein:
    def __init__(self, stop_token = 4, mapfold = MAPPING_FOLDER, maxSampleLength = 512,
                 verbose = False, dropRate = 0.2, seqonly = False, noflipseq = False):

        self.stop_token = stop_token
        self.maxSampleLength = maxSampleLength
        self.verbose = verbose
        self.dropRate = dropRate
        self.seqonly = seqonly
        self.noflipseq = noflipseq
        self.tokenizer = Tokenizer()
        vocab_manager = VocabularyManager()
        self.oneEncoderLength = vocab_manager.vocab_size -1

    def transformSeq(self, seq, prob = 0.2):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>(1-prob):
            seq = seq[::-1]
        return seq

    def transformKwSet(self, kws, drop = 0.2):
        """
        Filter kws, dropout, and replace with lineage (including term at end)
        """
        for kw in kws:
            if np.random.random()<drop:
                kws.remove(kw)
        for i in range(len(kws)):
            kws[i] = self.tokenizer.kw_to_ctrl_idx[kws[i]]
        return kws

    def transformSample(self, proteinDict):
        """
        Function to transform/augment a sample.
        Padding with all zeros
        Returns an encoded sample (taxa's,kw's,sequence) and the existence level to multiply weights
        """

        existence = 1

        kws = self.transformKwSet(proteinDict['kw'], drop = self.dropRate)

        if proteinDict['ex'] in [4, 5]:
            existence += 1

        seq = self.transformSeq(proteinDict['seq'])

        seq = list(self.tokenizer.aa_to_ctrl_idx[seq[i]] for i in range(len(seq)))

        seq = np.array(kws + seq + [self.stop_token]).astype(int)

        thePadIndex = len(seq)

        encodedSample = np.full(self.maxSampleLength, self.oneEncoderLength, dtype = int)

        encodedSample[:len(seq)] = seq

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
    with open(DATA_PATH + DATA_FILE, 'rb') as handle:
        test_chunk = pickle.load(handle)

    obj = transformProtein(verbose=False, dropRate = 0.0, seqonly = False)

    for key in test_chunk:
        encodedSample, existence, thePadIndex = obj.transformSample(test_chunk[key])
        print('max sample len', obj.maxSampleLength)
        print('encodedSample: ', encodedSample)
        print('existence: ', existence)
        print('thePadIndex', thePadIndex)
        print('encodedSample + pad index: ', encodedSample[:thePadIndex])
        print('encodedSample - pad index: ', encodedSample[thePadIndex:])
        break
