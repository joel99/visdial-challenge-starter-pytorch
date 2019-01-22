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

        # fusion layer
        fusion_size = config["img_feature_size"] + config["lstm_hidden_size"] * 2
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

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
        img = img.view(batch_size, 1, self.config["img_feature_size"])
        img = img.repeat(1, num_rounds, 1)
        img = img.view(batch_size * num_rounds, self.config["img_feature_size"])

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)
        ques_embed = self.ques_rnn(ques_embed, batch["ques_len"])
        # ques_embed = ques_embed.view(batch_size * num_rounds, 1, -1)
        # ques_embed = ques_embed.repeat()

        # embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed = self.word_embed(hist)
        hist_embed = self.hist_rnn(hist_embed, batch["hist_len"])

        fused_vector = torch.cat((img, ques_embed, hist_embed), 1)
        fused_vector = self.dropout(fused_vector)

        # Convert fused_vector to attention
        att_hidden_size = 512
        lstm_size = self.config["lstm_hidden_size"]
        # ReLU for nonlinearity, as per modified VQA implementation
        self.att_fc = nn.Sequential(nn.Linear(fused_vector.shape[1], att_hidden_size), nn.ReLU())
        self.att_weights = weight_norm(nn.Linear(att_hidden_size, 1), dim=None) # Understand this one

        att_vector = self.att_fc(fused_vector)
        att_logits = self.att_weights(att_vector)
        att_weights = nn.functional.softmax(att_logits, 1)
        att_img = (att_weights * img).sum(1)

        # Match question and attended image dimensions with FC
        ques_net = nn.Sequential(nn.Linear(lstm_size, lstm_size), nn.ReLU())
        img_net = nn.Sequential(nn.Linear(self.config["img_feature_size"], lstm_size), nn.ReLU())

        ques_repr = ques_net(ques_embed)
        img_repr = img_net(att_img)

        att_question = img_repr * ques_repr
        embed_fc = nn.Linear(lstm_size, lstm_size)
        bu_embedding = torch.tanh(embed_fc(att_question))
        bu_embedding = bu_embedding.view(batch_size, num_rounds, -1)
        return bu_embedding