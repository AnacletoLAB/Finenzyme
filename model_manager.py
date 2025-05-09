'''
This module handles 
1) Progen vocabulary loading
2) definition of the first and last layers of ProGen: TiedEmbeddingsSoftmax
3) loading the proGen model
'''

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import pytorch_transformer
from torch.cuda.amp import autocast

class VocabularyManager:
    def __init__(self,  vocab_loc = 'mapping_files/vocab.txt'):
        with open(vocab_loc, encoding='utf-8') as file:
            self.vocab = file.read().split('\n')[:-1]
        self.vocab = list(map(lambda x: x.split(' ')[0], self.vocab))
        self.vocab_size = len(self.vocab) 

class TiedEmbeddingSoftmax(torch.nn.Module):
    def __init__(self, embedding_size=1280, **kwargs):
        super(TiedEmbeddingSoftmax, self).__init__()
        vocab_manager = VocabularyManager()
        vocab_size = vocab_manager.vocab_size
        self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
        self.b = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs, embed=True):
        with autocast(enabled=False):
            if embed:
                return torch.nn.functional.embedding(inputs, self.w)
            else:
                return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
    def __init__(self):
        super(CTRLmodel,self).__init__()
        self.tied_embedding_softmax = TiedEmbeddingSoftmax()
        self.decoder = pytorch_transformer.Decoder()

    def forward(self, inputs):
        with autocast(enabled=False):
            x = self.tied_embedding_softmax(inputs, embed=True)
            x = self.decoder(x)
            x = self.tied_embedding_softmax(x, embed=False)
        return x

    def loadCheckpoint(self, model_path, num_layers = 36):
        if os.path.exists(model_path):
            print('Found PyTorch checkpoint at ', model_path)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            #for value in checkpoint:
                #print(len(checkpoint[value]))
            try:
                self.tied_embedding_softmax.load_state_dict({
                    'w': checkpoint.pop('tied_embedding_softmax.w', None),
                    'b': checkpoint.pop('tied_embedding_softmax.b', None)
                })
            except:
                print('Error during loading.')
                #torch.save(checkpoint['model_state_dict'], model_path)
                #checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            #self.tied_embedding_softmax.load_state_dict({
            #        'w': checkpoint.pop('tied_embedding_softmax.w', None),
            #        'b': checkpoint.pop('tied_embedding_softmax.b', None)
            #    })
            try:
                self.decoder.load_state_dict({key.replace("encoder.", ""): value for key, value in checkpoint.items()})
            except Exception:
                self.decoder.load_state_dict({key.replace("decoder.", ""): value for key, value in checkpoint.items()})
        else:
            print('Could not find PyTorch checkpoint')
            sys.exit()