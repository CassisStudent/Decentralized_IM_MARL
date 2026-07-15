import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DeltaWorldModelsEnsemble(nn.Module):
    def __init__(self, obs_dim, combined_action_dim, latent_dim, hidden_dim=64, n_agents=1):
        super(DeltaWorldModelsEnsemble, self).__init__()
        
        # Inputs: RNN-state (state_dim) + action
        input_dim = obs_dim + combined_action_dim
        self.latent_dim = latent_dim
        
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim)
            ) for _ in range(5)
        ])

    def forward(self, obs, all_pred_actions_one_hot):
        inputs = th.cat([obs, all_pred_actions_one_hot], dim=-1)
        
        predictions = th.stack([model(inputs) for model in self.models], dim=0)
        return predictions

    def get_intrinsic_reward(self, obs, all_pred_actions_one_hot):
        with th.no_grad():
            predictions = self.forward(obs, all_pred_actions_one_hot)
            disagreement = predictions.var(dim=0).mean(dim=-1) 
            return disagreement
