import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import numpy as np
import pickle
import argparse
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
parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for dataloader')
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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0002) #lr, betas
        lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        
        self.transformFull = transformProtein(stop_token = 7)
        self.writer = SummaryWriter()

    def train(self, num_epochs):
        self.model.train()
        GPU = torch.cuda.is_available()
        iter_num = 0
        for epoch in range(num_epochs):
            loss_epoch = 0.0
            num_epoch = 0

            name = 'ec_5'
            print('Training on family: ', name)
            path_training = "data_enzymes_classes/all_families_data/training_"+name+ ".p"
            full_dataset = ProteinDataset(path_training, firstAAidx = self.firstAAidx, transformFull = self.transformFull)
            dataloader = DataLoader(full_dataset, shuffle = True, batch_size = self.batch_size,
                                        num_workers = self.num_workers, pin_memory = False) #TODO pinmem?

            samples_num_epoch = 0
            loss_accumulator = 0.0
            for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):

                if GPU:
                    sample = sample.cuda()
                    labels = labels.cuda()
                    padIndex = padIndex.cuda()
                    begAAindex = begAAindex.cuda()
                    existence = existence.cuda()
                else:
                    raise ValueError('No GPU available (not going to happen)')
                
                self.optimizer.zero_grad()
                output = self.model(sample)
                loss = self.criterion(output.permute(0,2,1), labels)
                loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.scheduler.step()

                iter_num += 1
                self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)  

                samples_num_epoch += sample.shape[0]
                loss_accumulator += loss.item()

                loss_epoch += loss.item()
                num_epoch += sample.shape[0]

                
                if iter_num %10 == 0:
                    print('epoch: ', epoch+1)
                    print('loss within epoch: ', loss_accumulator/samples_num_epoch)
                    samples_num_epoch = 0
                    loss_accumulator = 0.0



            print('epoch: ', epoch + 1)
            print('loss_epoch: ', loss_epoch/num_epoch, epoch + 1)
            self.writer.add_scalar('Loss_epoch',loss_epoch/num_epoch, epoch + 1)

            if epoch + 1 == 4:
                print('Saving checkpoint..')
                torch.save(self.model.state_dict(),'ckpt/ec_5_4epochs_flip_LR02_16batch.pth')

                                   
        print('Training ended. saving last checkpoint')
        print('Saving last checkpoint..')
        torch.save(self.model.state_dict(),'ckpt/ec_5_6epochs_flip_doubleLR02_16batch.pth')


training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_manager.vocab_size,
                   model_dir = args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)
