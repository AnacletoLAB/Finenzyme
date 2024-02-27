import os
import torch
import numpy as np
import pickle
import argparse
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from torch.utils.tensorboard import SummaryWriter
from model_manager import CTRLmodel
from model_manager import VocabularyManager
from transformProtein import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='Code to train ProGen')
parser.add_argument('--model_dir', type =str, default='ckpt/',
                                        help='location of training model checkpoint')
parser.add_argument('--model_path', type=str, default='ckpt/pretrain_progen_full.pth', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')
parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=511,
                                        help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for dataloader')
parser.add_argument('--num_workers', type=int, default=2, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=100, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=10, help='save model checkpoint every X iterations')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

vocab_manager = VocabularyManager()
# sequence length to use for transfomer
seq_length = args.sequence_len

# load the model
model = CTRLmodel()
model.loadCheckpoint(model_path=args.model_path)

# freeze all weights except embedding
# for p in model.parameters():
#    p.requires_grad=False
model.tied_embedding_softmax.w.requires_grad=True
model.tied_embedding_softmax.b.requires_grad=True

if torch.cuda.is_available():
    model = model.cuda()
    print('previous checkpoint loaded in GPU')
else:
    print('previous checkpoint loaded')

class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26 # Assuming that the pad token is the last token and AAs are at the end
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001) #lr, betas
        lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        
        self.transformFull = transformProtein()
        self.writer = SummaryWriter()

    def train(self, num_epochs):
        self.model.train()
        GPU = torch.cuda.is_available()
        iter_num = 0
        for epoch in range(num_epochs):
            loss_e = 0.0
            num_e = 0
            
            for chunknum in range(10):
                pklpath = 'data_halogenase/chunks/'
                pklpath = pklpath + 'train' + str(chunknum) + '.p'
                chunk_dataset = ProteinDataset(pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull)
                dataloader = DataLoader(chunk_dataset, shuffle = True, batch_size = self.batch_size,
                                        num_workers = self.num_workers, pin_memory = False) #TODO pinmem?
        
                for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                    if GPU:
                        sample = sample.cuda()
                        labels = labels.cuda()
                        padIndex = padIndex.cuda()
                        begAAindex = begAAindex.cuda()
                        existence = existence.cuda()
                    self.optimizer.zero_grad()
                    output = self.model(sample)
                    #pdb.set_trace()
                    loss = self.criterion(output.permute(0,2,1), labels)
                    loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    self.scheduler.step()
                    loss_e += loss.item()
                    num_e += sample.shape[0]
                    iter_num += 1
                    self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)  

                print('epoch: ', epoch)
                print('chunknum: ', chunknum)
                print('self.save_iter: ', self.save_iter)
                if epoch == 2:
                    if (chunknum+1)%self.save_iter==0:
                        print('Saving checkpoint..')
                        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':i,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss,
                                   }, (self.model_dir + 'epoch' + str(epoch+1) + '_chunk' + str(chunknum+1) + '.pth'))
                loss_e/=num_e
                print('loss_e: ', loss_e)
            print('epoch: ', epoch)
            print('loss_e: ', loss_e)
            self.writer.add_scalar('Loss_epoch',loss_e, epoch)
        print('Training ended. saving last checkpoint')
        print('Saving last checkpoint..')
        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    }, 'ckpt/model_TEST.pth')


training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_manager.vocab_size,
                   model_dir = args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)
