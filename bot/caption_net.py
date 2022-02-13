import torch.nn as nn
import torch.nn.functional as F


class CaptionNet(nn.Module):
    def __init__(self, cnn_feature_size, vocab_size, embedding_size,
                 hidden_size, num_layers, dropout, padding_idx):
        super(self.__class__, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.hidden_projection = nn.Linear(cnn_feature_size,
                                           num_layers * hidden_size)
        self.cell_projection = nn.Linear(cnn_feature_size,
                                         num_layers * hidden_size)
        self.embeddings = nn.Embedding(vocab_size, embedding_size,
                                       padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_vectors, captions_ix):
        embedded = self.dropout(self.embeddings(captions_ix))
        # embedded = [batch size, seq len, embedding size]

        hidden = F.relu(self.hidden_projection(image_vectors))
        # hidden = [batch size, num layers * hidden size]
        hidden = hidden.reshape(-1, self.num_layers, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        # hidden = [num layers, batch size, hidden size]

        cell = F.relu(self.cell_projection(image_vectors))
        # cell = [batch size, num layers * hidden size]
        cell = cell.reshape(-1, self.num_layers, self.hidden_size)
        cell = cell.permute(1, 0, 2).contiguous()
        # cell = [num layers, batch size, hidden size]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [batch size, seq len, hidden size]

        logits = self.out(output)
        # logits = [batch size, seq len, vocab size]

        return logits
