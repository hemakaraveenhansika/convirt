import torch

# from models.resnet_clr import ResNetSimCLR
from models.model import ModelCLR
from models.model import DecoderRNN
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer
# from transformers import BertTokenizer

from models.tokenization import BertTokenizer

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        # shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
        shutil.copy('./config.yaml', '/kaggle/working/config.yaml')

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)#, do_lower_case=config['model_bert']['do_lower_case'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        #Dataloaders
        train_loader, valid_loader = self.dataset.get_train_data_loaders()

        #Model Resnet Initialize
        model = ModelCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=len(train_loader), 
                                                                eta_min=0, 
                                                                last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)


        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        #Checkpoint folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        ##decoder
        print('\nload decoder...')
        embed_size = 256
        hidden_size = 100
        num_layers = 1
        num_epochs = 4
        print_every = 150
        save_every = 1
        vocab_size = len(self.tokenizer.vocab)
        print(vocab_size)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(self.device)
        decoder.zero_grad()

        print(f'Training...')

        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for xis, xls in tqdm(train_loader):

                optimizer.zero_grad()
                # optimizer_bert.zero_grad()
                # print("\nbefor tokenizer")
                # print(xls)
                xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)
                # xls = self.tokenizer(xls, return_tensors="pt")
                # print("\nafter tokenizer")
                # print(xls)

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                # get the representations and the projections
                zis, zls = model(xis, xls)  # [N,C]

                ##decoder
                output = decoder(zis, xls)
                print("lstm output")
                print(output)

                # get the representations and the projections
                # zls = model_bert(xls)  # [N,C]
                # zls = xls
                # normalize projection feature vectors

                loss = self.nt_xent_criterion(zis, zls)

                # loss = self._step(model_res, model_bert, xis, xls, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                # optimizer_bert.step()
                n_iter += 1
                
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    # torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    torch.save(model.state_dict(), '/kaggle/working/model.pth')
                    print("save the model checkpoint in /kaggle/working/model.pth")
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        print("Training has finished...")
        # self.test()

    def test(self):
        print("\n\nTesting has started...")
        with torch.no_grad():  # turn off gradients computation
            # Dataloaders
            test_loader = self.dataset.get_test_data_loaders()

            # Model Resnet Initialize
            model = ModelCLR(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights_test(model)
            print("Testing: Loaded pre-trained model with success.")
            print(f'Testing...')

            for xis in tqdm(test_loader):

                xis = xis.to(self.device)
                zis = model(xis, None)  # [N]
                print("\n zis - v, Testing")
                print(zis)

        print("Testing has finished...")


    def _load_pre_trained_weights(self, model):
        try:
            print('search model.pth')
            # checkpoints_folder = os.path.join('/kaggle/input/', self.config['fine_tune_from'], '/checkpoints/')
            print('/kaggle/input/convirt-epoch-10/checkpoints/model.pth')
            # state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            state_dict = torch.load('/kaggle/input/convirt-epoch-10/checkpoints/model.pth')
            print('search model.pth done')
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _load_pre_trained_weights_test(self, model):
        try:
            save_path = '/kaggle/working/model.pth'
            state_dict = torch.load(save_path)
            model.load_state_dict(state_dict)

        except FileNotFoundError:
            print("Testing: Pre-trained weights not found.")

        return model

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():   #turn off gradients computation
            model.eval()
            # model_bert.eval()
            valid_loss = 0.0
            counter = 0
            print(f'Validation step')
            for xis, xls in tqdm(valid_loader):

                xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)
                # xls = self.tokenizer(xls, return_tensors="pt")

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                # get the representations and the projections
                zis, zls = model(xis, xls)  # [N,C]

                loss = self.nt_xent_criterion(zis, zls)

                valid_loss += loss.item()
                counter += 1

            valid_loss /= counter
        model.train()
        # model_bert.train()
        print("valid_loss : ", valid_loss)
        return valid_loss