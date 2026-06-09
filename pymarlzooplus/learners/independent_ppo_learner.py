# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from pymarlzooplus.components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
from torch.distributions import Categorical

from pymarlzooplus.modules.critics import DECENTRALIZED_REGISTRY as critic_registry
from pymarlzooplus.modules.agents import REGISTRY as actor_registry
from pymarlzooplus.components.action_selectors import REGISTRY as action_registry

from pymarlzooplus.components.standarize_stream import RunningMeanStd


class IndependentPPOLearner:
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
        self.world_model_ensemble = WorldModelsEnsemble(
            state_dim=args.hidden_dim,  # Jouw GRU rnn-state grootte (bijv 64)
            action_dim=act_dimension,   # Aantal mogelijke acties (bijv 5)
            latent_dim=obs_dimension,   # Wat we voorspellen (de volgende observatie/latent)
        ).to(self.device)
        
        self.world_model_optimiser = Adam(self.world_model.parameters(), lr=args.lr)


            
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
    
    def update4(self, obs, actions, old_logprobs, old_values, rewards, dones):
        device = next(self.actor.parameters()).device
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        old_values = old_values.to(self.device)
        dones = dones.to(self.device)
        
        buffer_size, max_seq_length = rewards.shape
        
        # ----------------------------
        # MASK
        # ----------------------------
        mask = th.ones_like(dones, device=device)
        mask[:, 1:] = (1 - dones[:, :-1]).cumprod(dim=1)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ----------------------------
        # RETURNS (per episode)
        # ----------------------------
        target_returns = self.compute_nstep_returns_vectorized(
            rewards,
            old_values,
            mask,
            self.args.q_nstep
        )
       
        advantages = (target_returns - old_values).detach()
        advantages = advantages * mask

        if self.args.standardise_advantages:
            valid = advantages[mask > 0]
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-8)
            advantages = advantages * mask
        
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
            
            grad_norm_actor = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()
            
            #optimise critic
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            """
            print("reward:", rewards.mean().item())
            print("actor loss:", actor_loss.item())
            print("critic loss:", critic_loss.item())
            print("entropy:", entropy_loss.item())
            print(self.hidden_state.abs().mean().item())
            print(ratio.mean().item())
            """
            # Print alleen de eerste en de laatste epoch om het overzichtelijk te houden
            if k == 0 or k == self.args.epochs - 1:
                with th.no_grad():
                    # Bereken bruikbare metrieken voor je overzicht
                    mean_ratio = ratio.mean().item()
                    approx_kl = (ratio - 1 - th.log(ratio)).mean().item() # Maatstaf voor stabiliteit
                    
                print(f"[{self.device}] Epoch {k+1}/{self.args.epochs} | "
                      f"PG Loss: {pg_loss.item():.4f} | "
                      f"Actor Loss: {actor_loss.item():.4f} | "
                      f"Critic Loss: {critic_loss.item():.4f} | "
                      f"Entropy Loss: {entropy_loss.item():.4f} | "
                      f"Mean Ratio: {mean_ratio:.3f} | "
                      f"Approx KL: {approx_kl:.4f}")
        
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
            
            
        
        # Scheidingslijn na de volledige update van deze agent
        print(f"[{self.device}] Adv Mean: {advantages.mean().item():.4f} | Adv Std: {advantages.std().item():.4f}")
    
    def update3(self, obs, actions, old_logprobs, old_values, rewards, dones):
        device = next(self.actor.parameters()).device
        
        print("updating");
        buffer_size, max_seq_length = rewards.shape
        
        # ----------------------------
        # MASK
        # ----------------------------
        mask = th.ones_like(dones)
        mask[:, 1:] = (1 - dones[:, :-1]).cumprod(dim=1)

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ----------------------------
        # RETURNS (per episode)
        # ----------------------------
        target_returns = th.zeros_like(rewards)
        
        for b_index in range(buffer_size):
            target_returns[b_index] = self.compute_nstep_returns(
                rewards[b_index],
                old_values[b_index],
                mask[b_index],
                self.args.q_nstep
            )
            
        advantages = (target_returns - old_values).detach()
        advantages = advantages * mask

        if self.args.standardise_advantages:
            valid = advantages[mask > 0]
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-8)
            advantages = advantages * mask

        print(advantages.mean().item(), advantages.std().item())
        
        for k in range(self.args.epochs):            
            new_logprobs = th.zeros_like(old_logprobs, device=device)
            entropies = th.zeros_like(old_logprobs, device=device)
            new_values = th.zeros_like(old_values, device=device)

            #Deze obs[step] is van 1 agent een t. laat dat duidelijk zijn.
            self.init_hidden_buffer(buffer_size)

            for step in range(max_seq_length):
                obs_t = obs[:, step]
                actions_t = actions[:, step]
                
                new_logprob, entropy= self.get_logprobs(obs_t, actions_t)
                #v = self.critic(obs_t).squeeze(-1)

                new_logprobs[:, step] = new_logprob
                entropies[:, step] = entropy
                #new_values[:, step] = v
            
            v = self.critic(obs).squeeze(-1)
            new_values = v
            """
            if new_values.shape == target_returns.shape:
                print("jjipppieee het klopt")
                print(new_values)
                print(target_returns)
            else:
                print("help nee het werkt niet")
                print(new_values.shape)
                print(target_returns.shape)
            
            print("rewards")
            print(rewards)
            """
            #actor loss
            ratio = th.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            
            pg_loss = -th.min(surr1, surr2)
            pg_loss = (pg_loss * mask).sum() / mask.sum()
            
            entropy_loss = -(entropies * mask).sum() / mask.sum()
            actor_loss = pg_loss - self.args.entropy_coef * entropy_loss
            
            #critic loss
            td_error = (target_returns.detach() - new_values)
            masked_td_error = td_error * mask
            critic_loss = (masked_td_error ** 2).sum() / mask.sum()

            #pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()
            
            #optimise critic
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            """
            print("reward:", rewards.mean().item())
            print("actor loss:", actor_loss.item())
            print("critic loss:", critic_loss.item())
            print("entropy:", entropy_loss.item())
            print(self.hidden_state.abs().mean().item())
            print(ratio.mean().item())
            """
        
    def update2(self, obs, actions, old_logprobs, old_values, rewards, dones):
        device = next(self.actor.parameters()).device
        
        #obs = th.tensor(obs, dtype=th.float32, device=device)
        #actions = th.tensor(actions, dtype=th.long, device=device)
        #old_logprobs = th.tensor(old_logprobs, dtype=th.float32, device=device)
        #old_values = th.tensor(old_values, dtype=th.float32, device=device)
        #rewards = th.tensor(rewards, dtype=th.float32, device=device)
        #dones = th.tensor(dones, dtype=th.float32, device=device)
        
        buffer_size, max_seq_length = rewards.shape
        
        # ----------------------------
        # MASK
        # ----------------------------
        mask = th.ones_like(dones)
        mask = (1 - dones).cumprod(dim=1)
        #for b_index in range(buffer_size):
            #for t in range(1, max_seq_length):
            #    mask[b_index, t] = mask[b_index, t-1] * (1 - dones[b_index, t-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ----------------------------
        # RETURNS (per episode)
        # ----------------------------
        target_returns = th.zeros_like(rewards)
        
        for b_index in range(buffer_size):
            target_returns[b_index] = self.compute_nstep_returns(
                rewards[b_index],
                old_values[b_index],
                mask[b_index],
                self.args.q_nstep
            )
            
        advantages = (target_returns - old_values).detach()

        for k in range(self.args.epochs):            
            new_logprobs = th.zeros_like(old_logprobs, device=device)
            entropies = th.zeros_like(old_logprobs, device=device)
            new_values = th.zeros_like(old_values, device=device)

            #Deze obs[step] is van 1 agent een t. laat dat duidelijk zijn.
            for b_index in range(buffer_size):
                self.init_hidden()
                
                ep_length = int(mask[b_index].sum().item())
                
                for step in range(ep_length):
                    new_logprob, entropy= self.get_logprobs(obs[b_index, step], actions[b_index, step])
                    v = self.critic(obs[b_index, step])

                    new_logprobs[b_index, step] = new_logprob
                    entropies[b_index, step] = entropy
                    new_values[b_index, step] = v

            
            #actor loss
            ratio = th.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            
            pg_loss = -th.min(surr1, surr2)
            pg_loss = (pg_loss * mask).sum() / mask.sum()
            
            entropy_loss = -(entropies * mask).sum() / mask.sum()
            actor_loss = pg_loss - self.args.entropy_coef * entropy_loss
            
            #critic loss
            td_error = (target_returns.detach() - new_values)
            masked_td_error = td_error * mask
            critic_loss = (masked_td_error ** 2).sum() / mask.sum()

            #pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()
            
            #optimise critic
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
    
    def update(self, obs, actions, old_logprobs, old_values, rewards, dones):
        
        device = next(self.actor.parameters()).device
        
        obs = th.tensor(obs, dtype=th.float32, device=device)
        actions = th.tensor(actions, dtype=th.long, device=device)
        old_logprobs = th.tensor(old_logprobs, dtype=th.float32, device=device)
        old_values = th.tensor(old_values, dtype=th.float32, device=device)
        rewards = th.tensor(rewards, dtype=th.float32, device=device)
        dones = th.tensor(dones, dtype=th.float32, device=device)
        
        # Build proper mask
        mask = th.ones_like(dones)
        for t in range(1, len(dones)):
            mask[t] = mask[t-1] * (1 - dones[t-1])
        
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        target_returns = self.compute_nstep_returns(rewards, old_values, mask, self.args.q_nstep)
        advantages = (target_returns - old_values).detach()

        for k in range(self.args.epochs):
            self.init_hidden()
            
            new_logprobs = th.zeros(len(obs), device=device)
            entropies = th.zeros(len(obs), device=device)
            new_values = th.zeros(len(obs), device=device)

            #Deze obs[step] is van 1 agent een t. laat dat duidelijk zijn.
            for step in range(len(obs)):
                new_logprob, entropy= self.get_logprobs(obs[step], actions[step])
                v = self.critic(obs[step])
                
                new_logprobs[step] = new_logprob
                entropies[step] = entropy
                new_values[step] = v

            
            #actor loss
            ratio = th.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            
            pg_loss = -th.min(surr1, surr2)
            pg_loss = (pg_loss * mask).sum() / mask.sum()
            
            entropy_loss = -(entropies * mask).sum() / mask.sum()
            actor_loss = pg_loss - self.args.entropy_coef * entropy_loss
            
            #critic loss
            td_error = (target_returns.detach() - new_values)
            masked_td_error = td_error * mask
            critic_loss = (masked_td_error ** 2).sum() / mask.sum()

            #pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()
            
            #optimise critic
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            
            th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            
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

            # Bootstrap met de waarde van de critic (V) na n stappen
            bootstrap_idx = t + n_steps
            if bootstrap_idx < T:
                # Alleen bootstrappen als we daadwerkelijk n_steps vooruit konden én de env nog actief is
                bootstrap_condition = (mask[:, bootstrap_idx] == 1) & (last_step == n_steps)
                bootstrap_value = (gamma ** n_steps) * values[:, bootstrap_idx]
                
                # Voeg toe waar de conditie True is
                return_t += bootstrap_condition.float() * bootstrap_value

            returns[:, t] = return_t

        return returns

        

        

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        for k in range(self.args.epochs):

            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()

            # Calculate policy grad with mask
            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.actor_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
                self.args.target_update_interval_or_tau > 1 and
                (
                        (self.critic_training_steps - self.last_target_update_step) /
                        self.args.target_update_interval_or_tau >= 1.0
                )
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):

        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

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