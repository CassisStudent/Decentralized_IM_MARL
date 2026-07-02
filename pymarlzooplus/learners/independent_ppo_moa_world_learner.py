# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from pymarlzooplus.components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam, AdamW
from torch.distributions import Categorical

from pymarlzooplus.modules.critics import DECENTRALIZED_REGISTRY as critic_registry
from pymarlzooplus.modules.agents import REGISTRY as actor_registry
from pymarlzooplus.components.action_selectors import REGISTRY as action_registry

from pymarlzooplus.components.standarize_stream import RunningMeanStd
from pymarlzooplus.modules.world_model.world_models_ensemble_moa import WorldModelsEnsembleMOA
from pymarlzooplus.modules.world_model.moa import MOA_LSTM


class IndependentPPOWorldLearnerMOA:
    def __init__(self, obs_dimension, act_dimension, args, device="cuda:0"):#, policy, critic, args):
        self.args = args
        self.device = device if args.use_cuda else "cpu"
                
        self.actor = actor_registry[self.args.agent](obs_dimension, self.args).to(self.device) ## args.agent should BE rnnAgent.
        self.actor_params = list(self.actor.parameters())
        self.actor_optimiser = Adam(params=self.actor_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](obs_dimension, self.args).to(self.device)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)
        
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        #device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(1, ), device=self.device) # changed the shape from self.n_agents to 1. IMPORTANT TO KEEP IN MIND
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self.device)
        
        #We don't need it
        #self.action_selector = action_REGISTRY[args.action_selector](args)
        
        self.hidden_state = None
        total_action_dim = self.args.n_actions * self.args.n_agents
        
        self.world_model = WorldModelsEnsembleMOA(
            state_dim=args.hidden_dim,  # rnn-state
            combined_action_dim=total_action_dim,   # number of actions
            latent_dim=obs_dimension,   # the prediction of the next state
        ).to(self.device)
        
        self.world_model_optimiser = AdamW(
            self.world_model.parameters(),
            lr=3e-4, 
            weight_decay=1e-4
        )
        
        self.moa = MOA_LSTM(
            obs_dim=obs_dimension,
            my_action_dim=act_dimension,
            partner_action_dim=act_dimension, #LET OP. GAAT ERVAN UIT DAT ALLE AGENTS DEZELFDE ACTIONSPACE HEBBEN 
            n_partners=self.args.n_agents - 1
        ).to(self.device)
        
        self.moa_optimiser = Adam(params=self.moa.parameters(), lr=self.args.lr_moa)

    """        
    def select_action(self, obs, action=None, available_actions=None):
        #Not sure if this reshaping of the observation is necessary but let's see what happens        
        logits, self.hidden_state = self.actor(obs, self.hidden_state)

        if available_actions is not None:
            logits[available_actions == 0] = -1e10  # mask invalid actions
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
    
        log_prob = probs.log_prob(action)
        value = self.critic(obs)
        
        # misschien #action.item()?
        return action, log_prob, probs.entropy(), value
    """
    
    def select_action(self, obs, hidden_state, action=None, available_actions=None):
        obs = obs.to(self.device)
        hidden_state = hidden_state.to(self.device)
        
        logits, next_hidden_state = self.actor(obs, hidden_state)
        
        if available_actions is not None:
            logits[available_actions == 0] = -1e10  # mask invalid actions
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
    
        log_prob = probs.log_prob(action)
        value = self.critic(obs)
        
        # misschien #action.item()?
        return action, log_prob, probs.entropy(), value, next_hidden_state
    
    def select_action_logits(self, obs, hidden_state, action=None, available_actions=None):
        obs = obs.to(self.device)
        hidden_state = hidden_state.to(self.device)
        
        logits, next_hidden_state = self.actor(obs, hidden_state)
        
        if available_actions is not None:
            logits[available_actions == 0] = -1e10  # mask invalid actions
        
        action = th.argmax(logits, dim=-1).item()
    
        value = self.critic(obs)
        
        # misschien #action.item()?
        return action, value, next_hidden_state
    
    def get_logprobs(self, obs, action, available_actions=None):
        logits, self.hidden_state = self.actor(obs, self.hidden_state)
        if available_actions is not None:
            logits[available_actions == 0] = -1e10  # mask invalid actions
        
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy()
    

    def init_hidden(self):
        self.hidden_state = self.actor.init_hidden()
    
    def init_hidden_buffer(self, buffer_size):
        self.hidden_state = self.actor.init_hidden().expand(buffer_size, -1).contiguous()
    
    def update4(self, obs, actions, old_logprobs, old_values, rewards, dones, rnn_states, team_actions, t_environment):
        rnn_states = rnn_states.to(self.device)
        
        device = next(self.actor.parameters()).device
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        old_values = old_values.to(self.device)
        dones = dones.to(self.device)
        team_actions = team_actions.to(self.device)        
        
        buffer_size, max_seq_length = rewards.shape
        
        # ----------------------------
        # MASK
        # ----------------------------
        #print("dones,", dones)
        mask = th.ones_like(dones, device=device)
        mask[:, 1:] = (1 - dones[:, :-1]).cumprod(dim=1)
        
        #print("mask,", mask)
        # =====================================================================
        # STAP 1: WORLD MODEL UPDATE (before PPO Berekeningen)
        # =====================================================================
        action_dim = self.args.n_actions
        actions_one_hot = th.nn.functional.one_hot(actions.squeeze(-1).long(), num_classes=action_dim).float()
        
        ##===============================
        # STAP 1A; MOA UPDATE
        #================================
        team_actions_one_hot = th.nn.functional.one_hot(team_actions.long(), num_classes=action_dim).float()
        team_actions_one_hot_flat = team_actions_one_hot.view(buffer_size, max_seq_length, (self.args.n_agents - 1) * action_dim)
        
        all_actions_one_hot = th.cat([actions_one_hot, team_actions_one_hot_flat], dim=-1)
        
        moa_input_obs = obs[:, :-1]
        moa_input_actions = all_actions_one_hot[:, :-1]
        moa_target_partners = team_actions[:, 1:] # Wat doen de partners op t+1?
        
        

        # cut-off
        current_rnn_states = rnn_states[:, :-1]
        current_my_actions = actions_one_hot[:, :-1]
        all_current_actions = moa_input_actions
        next_obs_target = obs[:, 1:].detach()
        wm_mask = mask[:, :-1]
        """
        print("wm_mask " + str(wm_mask.shape))
        print("rnn_states " + str(rnn_states.shape))
        print("current_action " + str(current_actions.shape))
        print("current_rnn_states " + str(current_rnn_states.shape))
        """
        
        if wm_mask.dim() == 2:
            wm_mask_3d = wm_mask.unsqueeze(-1) # -> [32, 499, 1]
        else:
            wm_mask_3d = wm_mask
        
        
        # training world model ensemble
        for wm_epoch in range(self.args.wm_epochs):
            next_action_logits, _ = self.moa(moa_input_obs, moa_input_actions)
            
            moa_loss_flat = th.nn.functional.cross_entropy(
                next_action_logits.view(-1, action_dim),
                moa_target_partners.contiguous().view(-1).long(),
                reduction="none"
            ).view(buffer_size, max_seq_length - 1, self.args.n_agents - 1)
            
            moa_loss = (moa_loss_flat * wm_mask_3d).sum() / wm_mask_3d.sum()
            moa_loss = moa_loss * self.args.moa_loss_weight # scaling conform paper (bijv. 10.0). #DIT MOET NOG IN  DE TEST_ARGS KOMEN TE STAAN DAN....
            self.moa_optimiser.zero_grad()
            moa_loss.backward()
            th.nn.utils.clip_grad_norm_(self.moa.parameters(), max_norm=0.5) #VERANDER DE MAX_NORM NOG NAAR EEN  GOED GETAL?
            self.moa_optimiser.step()

            #predicted next actions for world_model;
            with th.no_grad():
                current_logits, _ = self.moa(moa_input_obs, moa_input_actions)
                moa_probs = th.nn.functional.softmax(current_logits, dim=-1)
                # Platpersen naar [Batch, Tijd-1, N_Partners * Action_Dim]
                moa_probs_flat = moa_probs.view(buffer_size, max_seq_length - 1, -1).detach()
                
            wm_combined_actions_input = th.cat([current_my_actions, moa_probs_flat], dim=-1)
            predictions = self.world_model(current_rnn_states, wm_combined_actions_input)
            
            #  MSE loss per model
            wm_loss = 0.0
            
            for model_pred in predictions:
                squared_errors = (model_pred - next_obs_target) ** 2
                mean_squared_errors = squared_errors.mean(dim=-1, keepdim=True) # Shape: [32, 499, 1]
                wm_loss += (mean_squared_errors * wm_mask_3d).sum() / wm_mask_3d.sum()
                #wm_loss += (squared_errors.sum(dim=-1, keepdim=True) * wm_mask).sum() / wm_mask.sum()
                
            wm_loss = wm_loss / 5.0 # 5.0 is the enumber of world models we use
            
            self.world_model_optimiser.zero_grad()
            wm_loss.backward()
            th.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimiser.step()
        
        # =====================================================================
        # STAP 2: BEREKEN EN NORMALISEER DE INTRINSIEKE REWARD
        # =====================================================================
        with th.no_grad():
            intrinsic_reward = self.world_model.get_intrinsic_reward(current_rnn_states, wm_combined_actions_input)
            
            # Normalisatie
            valid_rewards = intrinsic_reward[wm_mask.bool()]
            reward_std = valid_rewards.std() + 1e-8
            reward_mean = valid_rewards.mean()
            
            # Z-score normalisation
            intrinsic_reward_norm = (intrinsic_reward - reward_mean) / reward_std
            
            full_intrinsic = th.zeros_like(rewards)
            if t_environment > 200000:
                full_intrinsic[:, :-1] = intrinsic_reward_norm
        
        total_rewards = rewards + self.args.beta * full_intrinsic

        if self.args.standardise_rewards:
            self.rew_ms.update(total_rewards)
            total_rewards = (total_rewards  - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ----------------------------
        # RETURNS (per episode)
        # ----------------------------
        target_returns = self.compute_nstep_returns_vectorized(
            total_rewards,
            old_values,
            mask,
            self.args.q_nstep
        )
        """
        print("rewards mean per step:", rewards.mean())
        print("rewards sum per episode:", rewards.sum(dim=1).mean())
        print("target_returns mean:", target_returns.mean())
        """
        advantages = (target_returns - old_values).detach()
        advantages = advantages * mask

        if self.args.standardise_advantages:
            valid = advantages[mask > 0]
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-8)
            advantages = advantages * mask
        
        
        
        #print("obs.shape" + str(obs.shape))
        for k in range(self.args.epochs):            
            new_logprobs = th.zeros_like(old_logprobs, device=device)
            entropies = th.zeros_like(old_logprobs, device=device)
            new_values = th.zeros_like(old_values, device=device)

            #Deze obs[step] is van 1 agent een t. laat dat duidelijk zijn.
            #self.init_hidden_buffer(buffer_size)
            
            local_hidden = self.actor.init_hidden().expand(1, buffer_size, -1).contiguous().to(device)
            
            logits, _ = self.actor(obs, local_hidden)
            probs = Categorical(logits=logits)
            new_logprobs = probs.log_prob(actions.squeeze(-1))
            entropies = probs.entropy()
            
            v = self.critic(obs).squeeze(-1)
            new_values = v
          
            #actor loss
            ratio = th.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            
            pg_loss = -th.min(surr1, surr2)
            pg_loss = (pg_loss * mask).sum() / mask.sum()
            
            entropy_loss = (entropies * mask).sum() / mask.sum() #Let op: heb de "min" '-' weggehaald
            actor_loss = pg_loss - self.args.entropy_coef * entropy_loss
            
            #critic loss
            td_error = (target_returns.detach() - new_values)
            masked_td_error = td_error * mask
            critic_loss = (masked_td_error ** 2).sum() / mask.sum()

            #pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            """
            print("New values", new_values.mean(), new_values.std())
            print("target returns", target_returns.mean(), target_returns.std())
            """
            grad_norm_actor = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()
            
            #optimise critic
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

            # print first and last epoch
            if k == 0 or k == self.args.epochs - 1:
                """
                print("ratio mean", ratio.mean().item())
                print("ratio std", ratio.std().item())
                print("old logprob mean", old_logprobs.mean())
                print("new logprob mean", new_logprobs.mean())

                diff = (new_logprobs - old_logprobs)

                print("diff mean", diff.mean())
                print("diff abs mean", diff.abs().mean())
                print("diff max", diff.abs().max())
                """
                with th.no_grad():
                    mean_ratio = ratio.mean().item()
                    approx_kl = (ratio - 1 - th.log(ratio)).mean().item() # stability measure
                    
                    raw_intrinsic_mean = valid_rewards.mean().item()
                    raw_intrinsic_std = valid_rewards.std().item() + 1e-8
                    
                    current_wm_loss = wm_loss.item() #mean loss van de 5 models
                """    
                print(f"[{self.device}] Epoch {k+1}/{self.args.epochs} | "
                      f"PG Loss: {pg_loss.item():.4f} | "
                      f"WM Loss: {current_wm_loss:.4f} | "
                      f"Intrinsic Raw Mean: {raw_intrinsic_mean:.4f} | "
                      f"Intrinsic Raw Std: {raw_intrinsic_std:.4f} | "
                      f"Actor Loss: {actor_loss.item():.4f} | "
                      f"Critic Loss: {critic_loss.item():.4f} | "
                      f"Entropy Loss: {entropy_loss.item():.4f} | "
                      f"Mean Ratio: {mean_ratio:.3f} | "
                      f"Approx KL: {approx_kl:.4f}")
                """
            
        
        #---- for logging-------#
        with th.no_grad():
            self.last_pg_loss = pg_loss.item()
            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()
            self.last_entropy = entropy_loss.item()
            self.last_mean_ratio = mean_ratio
            self.last_grad_norm_actor = grad_norm_actor.item()

            self.last_kl = approx_kl if 'approx_kl' in locals() else 0.0
            
            self.last_mean_advantage = advantages.mean().item()
            self.last_std_advantage = advantages.std().item()
            
            var_y = th.var(target_returns)
    
            # Als de variantie 0 is (bijv. in de allereerste stap), is de score NaN
            if var_y == 0:
                self.last_explained_var = 0.0
            else:
                # Formule: 1 - Var(Return - Prediction) / Var(Return)
                # We gebruiken .detach() om te zorgen dat we geen gradients meetrekken
                residual_var = th.var(target_returns.detach() - new_values.detach())
                self.last_explained_var = (1 - residual_var / var_y).item()
            
            #-------- TODO----------
            # SAVE INTRINSIC REAWRDS LOGS
            #--------------------------
            self.last_wm_loss = current_wm_loss
            self.last_intrinsic_raw_mean = raw_intrinsic_mean
            self.last_intrinsic_raw_std = raw_intrinsic_std
            
            # Sla ook de bèta op, handig als je later met decay gaat testen
            self.last_beta = self.args.beta
            
            
            
            
        
        # Scheidingslijn na de volledige update van deze agent
        #print(f"[{self.device}] Adv Mean: {advantages.mean().item():.4f} | Adv Std: {advantages.std().item():.4f}")
    
            
    def compute_nstep_returns(self, rewards, values, mask, n_steps):
        """
        Gt(n)=∑k=0n−1γkRt+k+γnV(st+n)
        """
        T = rewards.shape[0]
        returns = th.zeros_like(rewards)
        
        for t in range(T):
            return_t = 0.0
            last_step = 0  # track how far we actually went
            
            #look 5 steps ahead from current 'state' of reward.
            for n_step in range(n_steps):
                if t + n_step >= T or mask[t + n_step] == 0:
                    break
                return_t += (self.args.gamma ** n_step) * rewards[t + n_step]
                last_step += 1  # how many valid steps we used
            
            # if for prevention of overflowing and index mismatch. Last return is values of state
            if t + n_steps < T and mask[t + n_steps] == 1 and last_step == n_steps:
                return_t += (self.args.gamma ** n_steps) * values[t + n_steps]
            
            returns[t] = return_t
        
        return returns

    def compute_nstep_returns_vectorized(self, rewards, values, mask, n_steps):
        """
        Gevectoriseerde versie van n-step returns voor 2D tensors [Batch, Tijd].
        Berekent alle omgevingen tegelijk op de GPU!
        """
        device = next(self.actor.parameters()).device
        B, T = rewards.shape
        returns = th.zeros_like(rewards, device=device)
        gamma = self.args.gamma

        # Pre-computen van de discount factoren (gamma^0, gamma^1, ..., gamma^(n-1))
        # Dit hoeven we maar één keer te doen in plaats van elke iteratie
        gammas = th.pow(gamma, th.arange(n_steps, dtype=th.float32, device=device))

        # We lopen alleen over de n_steps (bijv. 5 keer), NIET over de tijd of omgevingen!
        for t in range(T):
            return_t = th.zeros(B, device=device)
            valid_mask = th.ones(B, device=device)
            last_step = th.zeros(B, device=device)

            # Lookahead loop (maximaal n_steps groot, heel snel op GPU)
            for n_step in range(n_steps):
                time_idx = t + n_step
                if time_idx >= T:
                    break
                
                # Update het masker: als een omgeving stopt (mask==0), stopt de optelling voor die env
                valid_mask = valid_mask * mask[:, time_idx]
                
                # Voeg de discounted reward toe voor alle omgevingen tegelijk
                return_t += valid_mask * (gammas[n_step] * rewards[:, time_idx])
                last_step += valid_mask
            
            
            bootstrap_idx = t + n_steps
            
            if bootstrap_idx < T:
                bootstrap_mask = mask[:, bootstrap_idx]
                bootstrap_value = (gamma ** n_steps) * values[:, bootstrap_idx]
                return_t += bootstrap_mask * bootstrap_value

            else:
                bootstrap_mask = mask[:, -1]
                bootstrap_value = (gamma ** (T - t)) * values[:, -1]
                return_t += bootstrap_mask * bootstrap_value

                
            returns[:, t] = return_t

        return returns

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
    
    def cuda_new(self):
        self.actor.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path, agent_id):
        th.save(self.actor.state_dict(), "{}/actor_agent_{}.th".format(path, agent_id))
        th.save(self.critic.state_dict(), "{}/critic_agent_{}.th".format(path, agent_id))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt_agent_{}.th".format(path, agent_id))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt_agent_{}.th".format(path, agent_id))

    def load_models(self, path, agent_id):
        # Laad de unieke bestanden in op de JUISTE GPU (self.device) waar de agent leeft
        self.actor.load_state_dict(
            th.load("{}/actor_agent_{}.th".format(path, agent_id), map_location=self.device)
        )
        self.critic.load_state_dict(
            th.load("{}/critic_agent_{}.th".format(path, agent_id), map_location=self.device)
        )
        
        # Synchroniseer de target critic direct
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimiser.load_state_dict(
            th.load("{}/actor_opt_agent_{}.th".format(path, agent_id), map_location=self.device)
        )
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt_agent_{}.th".format(path, agent_id), map_location=self.device)
        )
