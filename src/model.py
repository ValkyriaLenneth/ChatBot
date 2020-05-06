# Seq2Seq model
'''
    1. Encoder
    2. Decoder
        2.1 Attention
    3. Loss function
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Encoder
class Encoder(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(Encoder, self).__init__()

        # the values
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # model
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, bidirectional=True)

    def forward(self, input_seq, seq_lenth, hidden=0):
        # input: max_length * batch
        embedded = self.embedding(input_seq)
        # embedded: max_length * batch * hidden_size
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lenth)
        output, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, self.hidden_size:] + output[:, :, :self.hidden_size]
        # output: max_length * batch * hidden_size
        # hidden: (n_layers*n_direction) * batch * hidden_size
        return output, hidden

# Define the Attention Layer
class GlobalAttention(nn.Module):

    def __init__(self, hidden_size, method='dot'):
        super(GlobalAttention, self).__init__()

        self.hidden_size = hidden_size
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError("please input correct method for attention score")
        if self.method == 'general':
            self.w_general = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.w_concat = nn.Linear(hidden_size * 2, hidden_size)
            self.v_concat = nn.parameter(torch.FloatTensor(hidden_size))

    def score_dot(self, hidden, encoder_output):
        # hidden: (n_layer*n_dire) * batch * hidden_size
        # encoder_output: max_length * batch * hidden_size
        return torch.sum(hidden * encoder_output, dim=2)

    def score_general(self, hidden, encoder_output):
        attn_term = self.w_general(encoder_output)
        # attn_term: max_length * batch * hidden_size
        return torch.sum(hidden * attn_term, dim=2)

    def score_concat(self, hidden, encoder_output):
        # hidden: (n_layer*n_dire) * batch * hidden_size
        hidden = hidden.expand(encoder_output.size(0), -1, -1)
        # hidden: max_length * batch * hidden_size
        attn_term = self.w_concat(torch.cat( (hidden, encoder_output), dim=2)).tanh()
        return torch.sum(self.v_concat * attn_term, dim=2)

    def forward(self, hidden, encoder_output):
        if self.method == 'dot':
            score = self.score_dot(hidden, encoder_output)
        elif self.method == 'general':
            score = self.score_general(hidden, encoder_output)
        elif self.method == 'concat':
            score = self.score_concat(hidden, encoder_output)
        # score: max_length * batch
        attn_weight = F.softmax(score, dim=0)
        attn_weight = attn_weight.transpose(0,1)
        # attn_weight: batch * max_length
        return attn_weight.unsqueeze(1)
        # attn_weight: batch * 1 * max_length

class Decoder(nn.Module):

    def __init__(self, embedding, attention_model, hidden_size, vocab_size, dropout, n_layers=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.attention_model = attention_model

        # model
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
        self.attention = GlobalAttention(hidden_size, attention_model)
        self.w_concat = nn.Linear(hidden_size * 2, hidden_size)
        self.w_softmax = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, encoder_output, last_hidden):
        # input_token: 1, batch
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        gru_output, hidden = self.gru(embedded, last_hidden)
        # gru_output: 1, batch, hidden_size
        # hidden: (n_layer * n_dire) * batch * hidden_size
        attention_weight = self.attention(gru_output, encoder_output)
        # attention_weight: batch * 1 * max_length
        # encoder_output: max_length * batch * hidden_size
        context = torch.bmm(attention_weight, encoder_output.transpose(0, 1))
        # context: batch * 1 * hidden_size
        concated = self.w_concat(torch.cat((gru_output.squeeze(0), context.squeeze(1)), dim=1)).tanh()
        # concated: batch * hidden_size
        output = F.softmax(self.w_softmax(concated), dim=1)
        # output: batch * vocab_size
        return output, hidden


##########################################################
# Define Masked Loss
''' 
    Since the length of decoder sequence is not static, the answer given by
    Decoder is not as the same length with real answer in Q/A pairs.
    For we could not compare the semantic directly, we had to compare the 
    words one by one.
    So we need to constrain decoder to generate answer in the same length 
    with correct answer, then compare each word.
    But the input batch is padded, it's not necessary to calculate losses
    of them. Hence we use mask matrix to remove losses.
'''
def maskNLLLoss(input, target, mask):
    # how many words in this batch, since mask is a biTensor
    # we can just sum all elements.
    nTotal = mask.sum()

    # target: (max_length, batch) -> (max_length*batch, 1)
    # crossEntropy: -log(max_length*batch, 1, 1)
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1,1)).squeeze(1))
    # mask and src need not to be in the same size, but must have same
    # nums of elements.
    # Get the mean loss of this batch
    masked = torch.tensor(mask, device=device, dtype=torch.bool)
    loss = crossEntropy.masked_select(masked).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
