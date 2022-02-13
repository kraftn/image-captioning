from torch import nn
import torch.nn.functional as F
import torch


class FasterRCNNEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input):
        x = self.backbone(input)
        enc_hid_size = x['pool'].size(1)
        averaged_x = F.adaptive_avg_pool2d(x['pool'], (1, 1)).reshape(-1, enc_hid_size)
        return list(x.values()), averaged_x, None


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dec_hid_size, num_layers,
                 dropout, attention_layer, skip_connections, padding_idx):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dec_hid_size = dec_hid_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.skip_connections = skip_connections

        self.embeddings = nn.Embedding(vocab_size, embedding_size,
                                       padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = attention_layer
        self.rnn = nn.LSTM(input_size=embedding_size + sum(attention_layer.enc_hid_sizes),
                           hidden_size=dec_hid_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        if skip_connections:
            self.out = nn.Linear(dec_hid_size + sum(attention_layer.enc_hid_sizes) + embedding_size,
                                 vocab_size)
        else:
            self.out = nn.Linear(dec_hid_size, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [num layers, batch size, dec hid size]
        # cell = [num layers, batch size, dec hid size]
        # encoder_outputs = num_attentions, [batch size, enc vectors number, enc hid size]

        embedded = self.dropout(self.embeddings(input))
        # embedded = [batch size, embedding size]

        context_vectors, attentions = self.attention(hidden[-1], encoder_outputs)
        context_vectors = self.dropout(context_vectors)
        # context_vectors = [batch size, sum(enc hid sizes)]
        # attentions = num attentions, [batch size, enc vectors number]
        concatenated = torch.cat((embedded, context_vectors), dim=-1).unsqueeze(1)
        # concatenated = [batch size, 1, embedding size + sum(enc hid sizes)]

        decoder_outputs, (hidden, cell) = self.rnn(concatenated, (hidden, cell))
        # decoder_outputs = [batch size, 1, dec hid size]
        # hidden = [num layers, batch size, dec hid size]
        # cell = [num layers, batch size, dec hid size]

        if self.skip_connections:
            triple = torch.cat((concatenated, decoder_outputs), dim=-1).squeeze(1)
            # triple = [batch size, embedding size + sum(enc hid sizes) + dec hid size]
            logits = self.out(triple)
        else:
            decoder_outputs = decoder_outputs.squeeze(1)
            # decoder_outputs = [batch size, dec hid size]
            logits = self.out(decoder_outputs)

        # logits = [batch size, vocab size]

        return logits, hidden, cell, attentions


class Attention(nn.Module):
    '''
    Класс для совместимости с моделями с одним attention
    '''

    def __init__(self, enc_hid_size, dec_hid_size):
        super().__init__()

        self.enc_hid_sizes = [enc_hid_size]
        self.dec_hid_size = dec_hid_size

        self.attn = nn.Linear(enc_hid_size + dec_hid_size, dec_hid_size)
        self.v = nn.Linear(dec_hid_size, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid size]
        # encoder_outputs = 1, [batch size, enc vectors number, enc hid size]

        encoder_outputs = encoder_outputs[0]
        # encoder_outputs = [batch size, enc vectors number, enc hid size]

        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        # hidden = [batch size, enc vectors number, dec hid size]
        concatenated = torch.cat((hidden, encoder_outputs), dim=-1)
        # concatenated = [batch size, enc vectors number, dec hid size + enc hid size]
        attention = torch.tanh(self.attn(concatenated))
        attention = F.softmax(self.v(attention), dim=1)
        # attention = [batch size, enc vectors number, 1]

        context_vectors = (attention * encoder_outputs).sum(dim=1)
        # context_vectors = [batch size, enc hid size]
        attention = attention.squeeze(-1)
        # attention = [batch size, enc vectors number]
        return context_vectors, [attention]


class MultipleAttention(nn.Module):
    def __init__(self, enc_hid_sizes, dec_hid_size):
        super().__init__()

        self.enc_hid_sizes = enc_hid_sizes
        self.dec_hid_size = dec_hid_size

        self.attn = nn.ModuleList(
            [nn.Linear(enc_hid_size + dec_hid_size, dec_hid_size) for enc_hid_size in enc_hid_sizes])
        self.v = nn.ModuleList([nn.Linear(dec_hid_size, 1) for _ in range(len(enc_hid_sizes))])

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid size]
        # encoder_outputs = num attentions, [batch size, enc vectors number, enc hid size]

        hidden = hidden.unsqueeze(1)
        # hidden = [batch size, 1, dec hid size]

        context_vectors = []
        attentions = []

        for encoder_output, attn, v in zip(encoder_outputs, self.attn, self.v):
            repeated_hidden = hidden.repeat(1, encoder_output.size(1), 1)
            # repeated_hidden = [batch size, enc vectors number, dec hid size]
            concatenated = torch.cat((repeated_hidden, encoder_output), dim=-1)
            # concatenated = [batch size, enc vectors number, dec hid size + enc hid size]
            attention = torch.tanh(attn(concatenated))
            attention = F.softmax(v(attention), dim=1)
            # attention = [batch size, enc vectors number, 1]

            context_vector = (attention * encoder_output).sum(dim=1)
            # context_vector = [batch size, enc hid size]
            attention = attention.squeeze(-1)
            # attention = [batch size, enc vectors number]

            context_vectors.append(context_vector)
            attentions.append(attention)

        context_vectors = torch.hstack(context_vectors)
        # context_vectors = [batch size, sum(enc hid sizes)]
        # attentions = num attentions, [batch size, enc vectors number]

        return context_vectors, attentions


class CaptionNetWithAttention(nn.Module):
    def __init__(self, encoder, decoder, init_hidden_cell, out_enc_hid_size,
                 dropout, sos, eos):
        super().__init__()

        self.init_hidden_cell = init_hidden_cell
        self.out_enc_hid_size = out_enc_hid_size
        self.sos = sos
        self.eos = eos

        self.encoder = encoder
        if init_hidden_cell:
            self.hidden_projection = nn.Linear(out_enc_hid_size,
                                               decoder.num_layers * decoder.dec_hid_size)
            self.cell_projection = nn.Linear(out_enc_hid_size,
                                             decoder.num_layers * decoder.dec_hid_size)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = decoder

    def forward(self, images, captions_ix, sample=None, max_len=100):
        # images = [batch size, 3, height, width]
        # captions_ix = [batch size, seq len]

        encoder_outputs, img_codes, _ = self.encoder(images)
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]
        # encoder_outputs = num attentions, [batch size, enc hid size, n_rows, n_cols]
        # img_codes = [batch size, out enc hid size]

        hidden, cell = self.compute_initial_hidden_cell(img_codes)
        # hidden = [num layers, batch size, dec hid size]
        # cell = [num layers, batch size, dec hid size]

        batch_size = images.size(0)

        feature_maps_size = []
        for i_output, encoder_output in enumerate(encoder_outputs):
            feature_maps_size.append(encoder_output.size()[-2:])
            encoder_output = encoder_output.reshape(batch_size, self.decoder.attention.enc_hid_sizes[i_output], -1)
            encoder_outputs[i_output] = encoder_output.permute(0, 2, 1)
        # encoder_outputs = num attentions, [batch size, enc vectors number, enc hid size]
        # feature_maps_size = num attentions

        if captions_ix is not None:
            return self.train_prediction(captions_ix, hidden, cell, encoder_outputs)
        else:
            decoded, total_attentions = self.inference_prediction(sample, hidden, cell,
                                                                  encoder_outputs,
                                                                  max_len)
            return decoded, total_attentions, feature_maps_size

    def train(self, mode=True):
        self.training = mode

        if self.init_hidden_cell:
            self.hidden_projection.train(mode)
            self.cell_projection.train(mode)
        self.decoder.train(mode)

        return self

    def compute_initial_hidden_cell(self, img_codes):
        # img_codes = [batch size, out enc hid size]

        if self.init_hidden_cell:
            hidden = F.relu(self.hidden_projection(img_codes))
            # hidden = [batch size, num layers * dec hid size]
            hidden = hidden.reshape(-1, self.decoder.num_layers, self.decoder.dec_hid_size)
            hidden = hidden.permute(1, 0, 2).contiguous()
            hidden = self.dropout(hidden)
            # hidden = [num layers, batch size, dec hid size]

            cell = F.relu(self.cell_projection(img_codes))
            # cell = [batch size, num layers * dec hid size]
            cell = cell.reshape(-1, self.decoder.num_layers, self.decoder.dec_hid_size)
            cell = cell.permute(1, 0, 2).contiguous()
            cell = self.dropout(cell)
            # cell = [num layers, batch size, dec hid size]
        else:
            batch_size = img_codes.size(0)
            device = 'cuda' if img_codes.is_cuda else 'cpu'

            hidden = torch.zeros(self.decoder.num_layers, batch_size,
                                 self.decoder.dec_hid_size, device=device)
            cell = torch.zeros(self.decoder.num_layers, batch_size,
                               self.decoder.dec_hid_size, device=device)
        return hidden, cell

    def train_prediction(self, captions_ix, hidden, cell, encoder_outputs):
        # captions_ix = [batch size, seq len]
        # hidden = [num layers, batch size, dec hid size]
        # cell = [num layers, batch size, dec hid size]
        # encoder_outputs = num attentions, [batch size, enc vectors number, enc hid size]

        batch_size = captions_ix.size(0)
        seq_len = captions_ix.size(1)
        device = 'cuda' if captions_ix.is_cuda else 'cpu'

        decoded = torch.zeros(batch_size, seq_len, self.decoder.vocab_size,
                              device=device)
        for i_seq in range(seq_len):
            logits, hidden, cell, attentions = self.decoder(captions_ix[:, i_seq],
                                                            hidden, cell,
                                                            encoder_outputs)
            # logits = [batch size, vocab size]
            decoded[:, i_seq, :] = logits
        return decoded

    def inference_prediction(self, sample, hidden, cell, encoder_outputs, max_len):
        # hidden = [num layers, batch size, dec hid size]
        # cell = [num layers, batch size, dec hid size]
        # encoder_outputs = num attentions, [batch size, enc vectors number, enc hid size]

        batch_size = encoder_outputs[0].size(0)
        device = 'cuda' if encoder_outputs[0].is_cuda else 'cpu'

        input = torch.full((batch_size,), self.sos, device=device)
        decoded = [input]
        continue_mask = torch.full((batch_size,), True, device=device)
        total_attentions = [[] for _ in range(len(self.decoder.attention.enc_hid_sizes))]

        while len(decoded) <= max_len and continue_mask.sum() != 0:
            logits, hidden, cell, attentions = self.decoder(input, hidden, cell,
                                                            encoder_outputs)
            # logits = [batch size, vocab size]
            # attentions = num attentions, [batch size, enc vectors number]

            input = sample(logits)
            # input = [batch size]
            decoded.append(input)

            continue_mask = continue_mask & (input != self.eos)
            # continue_mask = [batch size]
            for total_attention, attention in zip(total_attentions, attentions):
                total_attention.append(attention)

        decoded = torch.stack(decoded, dim=1)
        # decoded = [batch size, seq len]
        for i, total_attention in enumerate(total_attentions):
            total_attentions[i] = torch.stack(total_attention).permute(1, 0, 2)
        # total_attentions = num attentions, [batch size, seq len, enc vectors number]

        return decoded, total_attentions
