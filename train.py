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
import base64
import csv
# from transformers import BertTokenizer

# from models.tokenization import BertTokenizer

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

import matplotlib.pyplot as plt
import skimage.io as io

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
        # self.tokenizer = BertTokenizer

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
        embed_size = 512
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
                print("\nbefor tokenizer")
                print(list(xls))
                xls_1 = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)
                xls_2 = self.tokenizer(list(xls), return_tensors="tf", padding=True, truncation=self.truncation)
                # xls = self.tokenizer(xls, return_tensors="pt")
                # print("\nafter tokenizer")
                # print(xls_2)

                xis = xis.to(self.device)
                xls_1 = xls_1.to(self.device)

                # print("xis :", xis.ndim, xis.shape)
                # get the representations and the projections
                zis, zls = model(xis, xls_1)  # [N,C]
                # print("\nzis")
                # print(zis)
                print("zis zls:", zis.shape, zls.shape)

                #decoder
                output = decoder(zis, xls_2)
                # print("lstm output", output.ndim, output.shape)

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




        print("\n\nTesting has started...")
        with torch.no_grad():  # turn off gradients computation
            # Dataloaders
            test_loader = self.dataset.get_test_data_loaders()
            test_result = []
            # test_loader_iter = iter(test_loader)
            # processed_img, processed_id = next(test_loader_iter)

            # Model Resnet Initialize
            # model = ModelCLR(**self.config["model"]).to(self.device)
            # model = self._load_pre_trained_weights_test(model)
            # print("Testing: Loaded pre-trained model with success.")
            print(f'Testing...', len(test_loader))
            # print(processed_img)
            #
            # processed_img = processed_img.to(self.device)
            # processed_zis = model(processed_img, None)  # [N]
            #
            # processed_features = processed_zis.unsqueeze(1)
            # final_output = decoder.predict(processed_features, max_len=20)
            # print("\n zis -> final_output, Testing")
            # print(final_output)
            img_url = ''

            for x in range(20):

                #encode
                test_value = test_loader[x]
                xis = test_value[0]
                processed_id = test_value[1]
                xis = xis.to(self.device)
                xis = xis.unsqueeze(0)
                # print("\n\nxis :", xis.ndim, xis.shape)
                zis = model(xis, None)  # [N]
                # print(zis)
                # print("test_loader", x)
                #
                # # array to b64encode
                # temp = zis.cpu().numpy()
                # base64_encode = base64.b64encode(temp)
                #
                # #b64encode to array
                # decodebytes = base64.decodebytes(base64_encode)
                # decodearray = np.frombuffer(decodebytes, dtype=np.float32)
                #
                # name = processed_id.split('/')[6]
                # test_result.append([name, base64_encode])

                #decode
                zis = zis.unsqueeze(0)
                features = zis.unsqueeze(1)
                final_output = decoder.predict(features, max_len=20)
                print("\n zis -> final_output, Testing", processed_id)
                print(final_output)
                xls_final = self.tokenizer.decode(final_output)
                print("\n xls_final")
                print(xls_final)
            # print(test_result)

            # with open("/kaggle/working/result.csv", 'w') as csvfile:
            #     # creating a csv writer object
            #     csvwriter = csv.writer(csvfile)
            #     # writing the data rows
            #     csvwriter.writerows(test_result)


        # I = io.imread(img_url)
        # plt.imshow(I)
        print("Testing has finished...")
        # self.test(model, decoder)

    def test(self):
    # def test(self, model, decoder):
        print("\n\nTesting has started...")
        with torch.no_grad():  # turn off gradients computation
            # Dataloaders
            test_loader = self.dataset.get_test_data_loaders()

            # Model Resnet Initialize
            model = ModelCLR(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights_test(model)
            print("Testing: Loaded pre-trained model with success.")
            print(f'Testing...')

            txt = open("/kaggle/working/myfile.txt", "a")


            # xis = test_value[0]
            # xis = xis.to(self.device)
            #
            # xis = xis.unsqueeze(0)
            #
            # print("xis :", xis.ndim, xis.shape)
            # print(test_value[1], '\n')
            #
            # zis = model(xis, None)  # [N]
            # print(zis)
            # print("zis :", zis.ndim, zis.shape)


            for x in range(10):
                test_value = test_loader[x]
                xis = test_value[0]
                print(test_value[1])
                xis = xis.to(self.device)
                xis = xis.unsqueeze(0)

                zis = model(xis, None)  # [N]
                print(zis)
                print("zis :", zis.ndim, zis.shape)
                txt.write(np.array_str(zis.cpu().numpy()))

                # features = zis.unsqueeze(1)
                # final_output = decoder.predict(features, max_len=20)
                # print("\n zis -> final_output, Testing")
                # print(final_output)

            txt.close()

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