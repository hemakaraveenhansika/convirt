"""
Reference for BERT Sentence Embeddings method

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",

"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from transformers import AutoModel

# Create the BertClassfier class
class ModelCLR(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, freeze_layers, do_lower_case):
        super(ModelCLR, self).__init__()
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings

        # init Resnet
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(res_base_model)
        num_ftrs = resnet.fc.in_features
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        # projection MLP for ResNet Model
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, out_dim)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            # print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            # print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder(self, xis):
        h = self.res_features(xis)
        h = h.squeeze()

        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)

        return h, x

    def text_encoder(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model(**encoded_inputs)
        # print("text_encoder outputs")
        # print(outputs)

        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).half()
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb

    def forward(self, xis, encoded_inputs):
        # print("\nforward layer")
        # print("encoded_inputs:")
        # print(encoded_inputs)

        h, zis = self.image_encoder(xis)

        if not encoded_inputs is None:
            zls = self.text_encoder(encoded_inputs)
            # print("end image_encoder, text_encoder")
            # print("\n zis - v")
            # print(zis)
            # print("Type of every element:", zis.dtype)
            # print("Number of axes:", zis.ndim)
            # print("Shape of tensor:", zis.shape)
            # print("Elements along axis 0 of tensor:", zis.shape[0])
            # print("Elements along the last axis of tensor:", zis.shape[-1])
            #
            # print("\n zls - u")
            # print(zls)
            # print("Type of every element:", zls.dtype)
            # print("Number of axes:", zls.ndim)
            # print("Shape of tensor:", zls.shape)
            # print("Elements along axis 0 of tensor:", zls.shape[0])
            # print("Elements along the last axis of tensor:", zls.shape[-1])
            return zis, zls
        else:
            return zis

####################

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True
                            )
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.device = self._get_device()

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

    def forward(self, features, captions):
        # captions = captions[:, :-1]
        print("\ncaptions tensor")
        print(captions)
        captions = (captions['input_ids'].numpy())[0]
        print("\ncaptions numpy")
        print(captions)
        captions = torch.Tensor(captions).long()
        print("\ncaptions")
        # captions = torch.tensor(captions, dtype=torch.long)
        print(captions)
        captions = captions.to(self.device)

        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embedding(captions)

        fea2 = features.unsqueeze(dim=1)

        print("shapes", features.shape, fea2.shape, embeds.shape)
        print("\nfeatures")
        print(features)
        print("\nembeds")
        print(embeds)

        inputs = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.fc(lstm_out)
        return outputs

    def Predict(self, inputs, max_len=20):
        final_output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_idx = torch.max(outputs, dim=1)
            final_output.append(max_idx.cpu().numpy()[0].item())
            if (max_idx == 1 or len(final_output) >= 20):
                break

            inputs = self.word_embedding(max_idx)
            inputs = inputs.unsqueeze(1)
        return final_output