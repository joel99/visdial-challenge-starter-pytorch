import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

from visdialch.utils import DynamicRNN

# Attend to bottom up features using question and history
class BottomUpLateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary), config["word_embedding_size"], padding_idx=vocabulary.PAD_INDEX
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"]
        )
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        att_hidden_size = 512
        lstm_size = self.config["lstm_hidden_size"]

        # Attention
        self.att_fc = nn.Sequential(weight_norm(nn.Linear(lstm_size * 2 + self.config["img_feature_size"], att_hidden_size), dim=None), nn.ReLU())
        self.att_weights = weight_norm(nn.Linear(att_hidden_size, 1))
        
        # Embedders
        self.ques_net = nn.Sequential(weight_norm(nn.Linear(lstm_size, lstm_size), dim=None), nn.ReLU())
        self.img_net = nn.Sequential(weight_norm(nn.Linear(self.config["img_feature_size"], lstm_size), dim=None), nn.ReLU())

        # Now the fusion for embeddings (NOT for context)
        self.fusion = nn.Linear(lstm_size, lstm_size)

        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN bottom-up features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # shape: (batch_size, 10, max_sequence_length * 2 * 10), concatenated qa * 10 rounds
        hist = batch["hist"]
        # num_rounds = 10, even for test split (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()

        if img.dim() != 3:
            raise Exception("Expected bottom up image features with 3 dimensions, got {}".format(img.dim())) # Probably raise this outside of forward
        
        num_bu_feat = img.shape[1]

        # Create attention for each image feature
        
        # repeat image feature vectors to be provided for every round
        img = img.view(batch_size, 1, num_bu_feat, self.config["img_feature_size"])
        img = img.repeat(1, num_rounds, 1, 1)
        img = img.view(batch_size * num_rounds, num_bu_feat, self.config["img_feature_size"])

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_embed, batch["ques_len"])
        stacked_ques = ques_embed.unsqueeze(1).repeat(1, num_bu_feat, 1)

        # embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_rnn(hist_embed, batch["hist_len"])
        stacked_hist = hist_embed.unsqueeze(1).repeat(1, num_bu_feat, 1)

        fused_context = torch.cat((img, stacked_ques, stacked_hist), 2)
        fused_context = self.dropout(fused_context)

        # Convert fused_context to attention
        # ReLU for nonlinearity, as per modified VQA implementation
        att_vector = self.att_fc(fused_context)
        att_logits = self.att_weights(att_vector)
        att_weights = nn.functional.softmax(att_logits, 1)
        att_img = (att_weights * img).sum(1)

        # Match question and attended image dimensions with FC
        ques_repr = self.ques_net(ques_embed)
        img_repr = self.img_net(att_img)

        att_question = img_repr * ques_repr
        bu_embedding = nn.functional.tanh(self.fusion(att_question))
        bu_embedding = bu_embedding.view(batch_size, num_rounds, -1)
        return bu_embedding