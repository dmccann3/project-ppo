import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from net import Network


class PPO:

    def __init__(self, env, **hyperparameters):

        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        # init hyperparameters
        self._init_hyperparameters(hyperparameters)

        # init env
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        # init actor and critic networks
        self.actor = Network(self.num_states, self.num_actions)
        self.critic = Network(self.num_states, 1)

        # init optimizers for each network
        self.adam_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.adam_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        # init covariance matrix
        self.cov_var = torch.full(size=(self.num_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # init logging storage
        self.storage = {
            'dt': time.time_ns(),
            'episode': 0,
            'time_step': 0,
            'batch_len': [],
            'batch_rew': [],
            'actor_losses': [],
        }

        # init some lists for plotting
        self.training_returns = []
        self.training_losses_A = []
        self.training_losses_C = []


    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    
    def get_action(self, s):
        # Get mean from actor network
        mean = self.actor(s)

        # Make multi var dist to sample action from to decrease anomolies
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample action and get log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
    

    def get_rtg(self, reward_batch):
        # Make array to store rtgs
        rtg_batch = []

        for ep_rews in reversed(reward_batch):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                rtg_batch.insert(0, discounted_reward)

        rtg_batch = torch.tensor(rtg_batch, dtype=torch.float32)

        return rtg_batch
    
    
    def rollout(self):
        # Init all storage for batches
        state_batch = []
        action_batch = []
        log_prob_batch = []
        reward_batch = []
        rtg_batch = []
        lens_batch = []

        # Current timestep
        t = 0

        while t < self.timesteps_per_batch:
            ep_rew = []

            s, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Render if specified
                if self.render and (self.storage['episode'] % self.render_every_i == 0) and len(lens_batch) == 0:
                    self.env.render()

                # Increment t
                t += 1

                # Store current state
                state_batch.append(s)

                a, log_prob = self.get_action(s) 
                s, r, _, done,_ = self.env.step(a)                

                # Store action, log prob
                action_batch.append(a)
                log_prob_batch.append(log_prob)

                # Store reward to calc episode reward later
                ep_rew.append(r)

                # Check if episode is done
                if done:
                    break

            lens_batch.append(ep_t + 1)
            reward_batch.append(ep_rew)
            self.training_returns.append(ep_rew)


        # Collect all batch data as torch tensors
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.float32)
        log_prob_batch = torch.tensor(log_prob_batch, dtype=torch.float32)
        rtg_batch = self.get_rtg(reward_batch)

        # Store batch len and batch rew in global storage
        self.storage['batch_rew'] = reward_batch
        self.storage['batch_len'] = lens_batch

        return state_batch, action_batch, log_prob_batch, rtg_batch, lens_batch


    def evaluate(self, state_batch, action_batch):
        # Run critic network to get the values for each observation in the batch
        V = self.critic(state_batch).squeeze()

        # Claclualte the log probabilities 
        mean = self.actor(state_batch)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(action_batch)

        return V, log_probs


    def train(self, episodes):

        print(f'Training... Running {self.max_timesteps_per_episode} timesteps per episode', end='')
        print(f'{self.timesteps_per_batch} timesteps per batch for a total of {episodes} timesteps')

        episode = 0
        timestep = 0
        while episode < episodes:

            state_batch, action_batch, log_prob_batch, rtg_batch, lens_batch = self.rollout()

            # Get how many episodes we have covered so far
            episode += np.sum(lens_batch)

            # Increment the current timestep
            timestep += 1

            # Log the episodes taken so far
            self.storage['episode'] = episode
            self.storage['time_step'] = timestep

            # Calculate the advantage
            V, _ = self.evaluate(state_batch, action_batch)
            A_k = rtg_batch - V.detach()

            # Decrease the variance of the advantage (not in psuedocode but great for )
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update network
            for _ in range(self.n_updates_per_iteration):
                # Get V_phi
                V, cur_log_probs = self.evaluate(state_batch, action_batch)

                # Calculate ratio 
                # This is equivalent to the doing the ratio pi / pi_old (things cancel out if your write out the full equations for the agents)
                ratios = torch.exp(cur_log_probs - log_prob_batch)

                # Calculate surrogates losses (sur1 regular PG loss and sur2 the clipped portion)
                sur1 = ratios * A_k
                sur2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses
                actor_loss = (-torch.min(sur1, sur2)).mean()
                critic_loss = nn.MSELoss()(V, rtg_batch)

                # Calculate gradients and backprop for actor network
                self.adam_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.adam_actor.step()

                # Calculate gradients and backprop for critic network
                self.adam_critic.zero_grad()
                critic_loss.backward()
                self.adam_critic.step()

                # Log actor loss
                self.storage['actor_losses'].append(actor_loss.detach())
                self.training_losses_A.append(actor_loss.detach().numpy())
                self.training_losses_C.append(critic_loss.detach().numpy())

            # Print out the log
            self._log_summary()


            if episode % self.save_freq == 0:
                torch.save(self.actor.state_dict(), 'ppo_actor.pth')
                torch.save(self.critic.state_dict(), 'ppo_critic.pth')

    
    def _log_summary(self):
        # Calculate logging values. 
        delta_t = self.storage['dt']
        self.storage['dt'] = time.time_ns()
        delta_t = (self.storage['dt'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.storage['episode']
        i_so_far = self.storage['time_step']
        avg_ep_lens = np.mean(self.storage['batch_len'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.storage['batch_rew']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.storage['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.storage['batch_len'] = []
        self.storage['batch_rew'] = []
        self.storage['actor_losses'] = []

