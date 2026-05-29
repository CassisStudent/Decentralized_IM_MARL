# code adapted from https://github.com/wendelinboehmer/dcg
# and https://github.com/lich14/CDS/blob/main/CDS_GRF/modules/agents/rnn_agent.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from pymarlzooplus.utils.trainable_image_encoder import TrainableImageEncoder


class RNNAgentROME(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentROME, self).__init__()
        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.n_agents = args.n_agents

        # Use CNN to encode image observations
        self.is_image = False
        if isinstance(input_shape, tuple):  # image input
            self.cnn = TrainableImageEncoder(input_shape, args)
            input_shape = self.cnn.features_dim + input_shape[1]
            self.is_image = True

        assert self.is_image is False, "ROME does not support image obs for the time being!"
        if self.use_rnn is False:
            print("Running ROME Agent in MLP (Feedforward) mode without RNN!")
        
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.use_rnn is True:
            self.rnn = nn.GRU(
                input_size=args.hidden_dim,
                num_layers=1,
                hidden_size=args.hidden_dim,
                batch_first=True
            )
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        is_2d = (inputs.dim() == 2)
        
        if is_2d:
            # [Batch, Features] -> [Batch, 1, Features]
            inputs = inputs.unsqueeze(1)

        if self.is_image is True:
            inputs[0] = self.cnn(inputs[0])
            if len(inputs[1] > 0):
                inputs = th.concat(inputs, dim=1)
            else:
                inputs = inputs[0]

        bs = inputs.shape[0]
        epi_len = inputs.shape[1]
        num_feat = inputs.shape[2]
        inputs = inputs.reshape(bs * epi_len, num_feat)

        x = F.relu(self.fc1(inputs))
        
        if self.use_rnn:
            x = x.reshape(bs, epi_len, self.args.hidden_dim)

            h_in = hidden_state.reshape(1, bs, self.args.hidden_dim).contiguous()
            x, h = self.rnn(x, h_in)

            x = x.reshape(bs * epi_len, self.args.hidden_dim)
        else:
            x = F.relu(self.rnn(x))
            h = hidden_state
            
        q = self.fc2(x)
        q = q.reshape(bs, epi_len, self.args.n_actions)
        
        if is_2d:
            # [Batch, 1, Actions] -> [Batch, Actions]
            q = q.squeeze(1)

        return q, h

