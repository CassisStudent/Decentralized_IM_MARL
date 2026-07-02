import torch as th
import torch.nn as nn
import torch.nn.functional as F
"""
class ModelOfOtherAgents(nn.Module):
    def __init__(self, my_state_dim, my_action_dim, partner_action_dim, n_partners=3, hidden_dim=128):
        super().__init__()
        self.n_partners = n_partners
        self.partner_action_dim = partner_action_dim
        
        input_dim = my_state_dim + my_action_dim
        #3 * 5 = 15 logits
        output_dim = n_partners * partner_action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, rnn_state, my_action_one_hot):
        inputs = th.cat([rnn_state, my_action_one_hot], dim=-1)
        logits = self.net(inputs)
        
        # Reshape de logits naar [Batch, Tijd, n_partners, partner_action_dim]
        # Dit maakt het makkelijk om straks cross-entropy toe te passen per partner!
        B, T, _ = rnn_state.shape
        return logits.view(B, T, self.n_partners, self.partner_action_dim)
    

class MOA_Model(nn.Module):
    def __init__(self, obs_dim, my_action_dim, partner_action_dim, n_partners=3, hidden_dim=128):
        super().__init__()
        self.n_partners = n_partners
        self.partner_action_dim = partner_action_dim
        self.hidden_dim = hidden_dim
        
        # Invoer: Mijn ruwe observatie + Mijn Actie (one-hot)
        # We gebruiken de ruwe obs in plaats van de rnn_state voor stabiliteit!
        input_dim = obs_dim + my_action_dim
        output_dim = n_partners * partner_action_dim
        
        # Layer 1: Lineaire embedding (Exact zoals de fully connected layers uit de paper)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2: De LSTM met 128 of 128+ cells (Table 3 specificatie uit de paper)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Layer 3: Output logits voor alle partners tegelijk
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, my_obs, my_action_one_hot, hidden_state=None):
        #Inputs:
        #    my_obs:              [Batch, Tijd, Obs_Dim]
        #    my_action_one_hot:    [Batch, Tijd, My_Action_Dim]
        #    hidden_state:         Een tuple (h_0, c_0) voor de LSTM, standaard None voor complete 3D pass
        #Outputs:
        #    logits:              [Batch, Tijd, n_partners, partner_action_dim]
        B, T, _ = my_obs.shape
        
        # Concateneer de features over de laatste dimensie
        x = th.cat([my_obs, my_action_one_hot], dim=-1)
        
        # Flatten over batch en tijd voor de eerste lineaire laag
        x_flat = x.reshape(B * T, -1)
        h1 = F.relu(self.fc1(x_flat))
        
        # Terugshapen naar 3D voor de LSTM sequence pass
        h1_seq = h1.reshape(B, T, self.hidden_dim)
        
        # Jaag de complete tijdreeks door de LSTM (vlijmsnel op GPU)
        # out shape: [Batch, Tijd, Hidden_Dim]
        out, _ = self.lstm(h1_seq, hidden_state)
        
        # Flatten de output voor de finale lineaire laag
        out_flat = out.reshape(B * T, self.hidden_dim)
        logits_flat = self.fc2(out_flat)
        
        # Reshape naar de uiteindelijke 4D structuur: [Batch, Tijd, Partners, Acties]
        return logits_flat.view(B, T, self.n_partners, self.partner_action_dim)
"""

class MOA_LSTM(nn.Module):
    def __init__(self, obs_dim, my_action_dim, partner_action_dim, n_partners=3, cell_size=64):
        super(MOA_LSTM, self).__init__()
        self.n_partners = n_partners
        self.my_acttion_dim = my_action_dim
        self.partner_action_dim = partner_action_dim
        self.cell_size = cell_size

        # Concat_input dimensie: Mijn Obs + (Mijn Actie + Alle Partner Acties)
        # In de paper plakken ze alle one-hot acties van ALLE agents aan de observatie
        self.total_action_dim = my_action_dim + (n_partners * partner_action_dim)
        input_dim = obs_dim + self.total_action_dim

        # De LSTM laag (exact zoals tf.keras.layers.LSTM uit de paper)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cell_size,
            batch_first=True # Zorgt voor matching met jouw [Batch, Tijd, Features]
        )

        # De finale lineaire laag (exact zoals tf.keras.layers.Dense na de lstm)
        self.fc_out = nn.Linear(cell_size, n_partners * partner_action_dim)

    
    def forward(self, my_obs, all_actions_one_hot, hidden_state=None):
        """
        Inputs:
            my_obs:                [Batch, Tijd, Obs_Dim]
            all_actions_one_hot:   [Batch, Tijd, Total_Action_Dim] (Jou + Partners samen)
            hidden_state:          Tuple (h, c) voor de LSTM stappen, standaard None voor 3D pass
        Outputs:
            logits:                [Batch, Tijd, n_partners, partner_action_dim]
            next_hidden:           Tuple (h, c) voor de volgende stap
        """
        B, T, _ = my_obs.shape

        # 1. Keras Concateneer stap: [Batch, Tijd, Obs_Dim + Total_Action_Dim]
        concat_input = th.cat([my_obs, all_actions_one_hot], dim=-1)

        # 2. De LSTM Sequence pass
        # out shape: [Batch, Tijd, cell_size]
        out, next_hidden = self.lstm(concat_input, hidden_state)

        # 3. Keras Dense stap: projecteer cell_size naar de totale partner logits
        # Flatten over Batch*Tijd voor de lineaire laag
        logits_flat = self.fc_out(out.reshape(B * T, self.cell_size))

        # 4. Breng terug naar de overzichtelijke 4D Multi-Agent structuur
        logits = logits_flat.view(B, T, self.n_partners, self.partner_action_dim)

        #probabilites = F.softmax(logits, dim=-1)
        #return logits, next_hidden
        return logits, next_hidden
    
    
    def get_marginal_predictions(moa_model, current_obs, my_policy_probs, action_dim, all_my_actions):
        """
        Vertaald uit de Keras-functie van de auteurs.
        Berekent de marginale kansen van de partners gewogen naar jouw huidige beleid.
        """
        B, T, _ = current_obs.shape
        device = current_obs.device
        all_my_actions = th.eye(action_dim, device=device) # [5, 5]

        # Maak een mega-batch van alle 5 mogelijke acties die IK had kunnen kiezen
        obs_expanded = current_obs.unsqueeze(0).expand(action_dim, B, T, -1).reshape(action_dim * B, T, -1)
        my_actions_expanded = all_my_actions.view(action_dim, 1, 1, action_dim).expand(-1, B, T, -1).reshape(action_dim * B, T, -1)

        # Vraag de counterfactual logits aan de MOA: [Actions * Batch, Tijd, Partners, Acties_Partner]
        counterfactual_logits = forward(obs_expanded, my_actions_expanded)
        counterfactual_probs = F.softmax(counterfactual_logits, dim=-1)

        # Reshape terug naar [5 Eigen_Acties, Batch, Tijd, Partners, Acties_Partner]
        cf_probs_expanded = counterfactual_probs.view(action_dim, B, T, moa_model.n_partners, moa_model.partner_action_dim)

        # Weeg de uitkomsten met jouw ECHTE actie-kansen (my_policy_probs)
        # my_policy_probs shape: [Batch, Tijd, Eigen_Acties] -> reshape voor broadcasting
        weight_logits = my_policy_probs.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1) # [5, B, T, 1, 1]

        # Sommeer over de 5 eigen acties (dim=0) om de marginale kansen te krijgen
        marginal_probs = (cf_probs_expanded * weight_logits).sum(dim=0)
        return marginal_probs # Shape: [Batch, Tijd, Partners, Acties_Partner]


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class MOAWorldModelsEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, partner_action_dim, n_partners=3, latent_dim=128, hidden_dim=128):
        super(WorldModelsEnsemble, self).__init__()
        
        # Inputs: RNN-state (state_dim) + action
        total_partner_dim = n_partners * partner_action_dim
        input_dim = state_dim + action_dim + total_partner_dim
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

    def forward(self, rnn_state, action_one_hot, partner_actions_flattened):
        inputs = th.cat([rnn_state, action_one_hot, partner_actions_flattened], dim=-1)
        
        predictions = th.stack([model(inputs) for model in self.models], dim=0)
        return predictions

    def get_intrinsic_reward(self, rnn_state, action_one_hot, partner_actions_flattened):
        with th.no_grad():
            predictions = self.forward(rnn_state, action_one_hot, partner_actions_flattened)
            disagreement = predictions.var(dim=0).mean(dim=-1) 
            return disagreement