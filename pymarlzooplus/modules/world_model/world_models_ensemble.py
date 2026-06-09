import torch as th
import torch.nn as nn
import torch.nn.functional as F

class WorldModelsEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=128):
        super(WorldModelsEnsemble, self).__init__()
        
        # Inputs: RNN-state (state_dim) + action
        input_dim = state_dim + action_dim
        self.latent_dim = latent_dim
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for _ in range(5)
        ])

    def forward(self, rnn_state, action_one_hot):
        inputs = th.cat([rnn_state, action_one_hot], dim=-1)
        
        predictions = th.stack([model(inputs) for model in self.models], dim=0)
        return predictions

    def get_intrinsic_reward(self, rnn_state, action_one_hot):
        with th.no_grad():
            predictions = self.forward(rnn_state, action_one_hot)
            disagreement = predictions.var(dim=0).mean(dim=-1) 
            return disagreement
