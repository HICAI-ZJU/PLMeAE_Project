import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from fairseq.models import FairseqDecoder

def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)



class ProteinRegressionDecoder(FairseqDecoder):
    def __init__(self, alphabet, embed_dim):
        super().__init__(alphabet)
        self.alphabet = alphabet
        self.embed_dim = embed_dim
        self.compressor = nn.RNN(embed_dim, embed_dim, 1)
        #
        self.hidden_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
        #
        self.classifier = nn.Linear(embed_dim, 1)

    def forward_rnn(self, prev_result, tokens):
        output = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype = torch.float32
        ).cuda()
        hidden = prev_result[:, 0, :]
        for idx, output_sequence in enumerate(prev_result):
            input_shape = (len(tokens[idx])-1, 1, self.embed_dim)
            output[idx, :] = self.compressor(output_sequence[1:, :].view(input_shape), hidden[idx, :].view(1, 1, self.embed_dim))[0][-1, 0, :]
        return self.classifier(output)
    #
    def forward_mlp(self, prev_result, tokens):
        output = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype = torch.float32
        ).cuda()
        for idx, output_sequence in enumerate(prev_result):
            # import pdb; pdb.set_trace()
            output[idx, :] = torch.mean(F.relu(self.fc1(output_sequence[1:sum(tokens[idx].ne(self.alphabet.padding_idx))-2, :])), axis=0)
        return self.classifier(output)



    #
    def forward_mean(self, prev_result, tokens):
        embedding = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype = torch.float32
        ).cuda()
        for idx, output_sequence in enumerate(prev_result):
            embedding[idx, :] = torch.mean(output_sequence[1:sum(tokens[idx].ne(self.alphabet.padding_idx))-2, :], axis=0)
        return self.classifier(embedding)


    def forward_sum(self, prev_result, tokens):
        embedding = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype = torch.float32
        ).cuda()
        for idx, output_sequence in enumerate(prev_result):
            embedding[idx, :] = torch.sum(output_sequence[1:sum(tokens[idx].ne(self.alphabet.padding_idx))-2, :], axis=0)
        return self.classifier(embedding)

    


class ProteinLMDecoder(FairseqDecoder):
    def __init__(self, args, embed_dim, weight, alphabet):
        super().__init__(alphabet)
        self.args = args
        self.embed_dim = embed_dim
        self.alphabet = alphabet

        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(len(alphabet)))


    def forward(self, prev_result):
        x = self.dense(prev_result)
        x = gelu(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x