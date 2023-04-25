import gym
import sys
import torch
import eval
import plot
import time
import base64
from gym import wrappers

from ppo import PPO
from net import Network
from arguments import get_args



def train(env, hyperparameters, actor_model, critic_model):
    print(f'Training', flush=True)

    model = PPO(env=env, **hyperparameters)

    if actor_model != '' and critic_model != '':
        print(f'Loading in {actor_model} amd {critic_model} ...', flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print('Successfully loaded.', flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f'Training from scratch.', flush=True)

    model.train(episodes=1_000_000)
    plot.write_to_csv(model.training_returns, model.training_losses_A, model.training_losses_C, 'Data/training_rets4.csv', 'Data/training_lss_A4.csv', 'Data/training_lss_C4.csv')


def test(env, actor_model):

    print(f'Testing {actor_model}', flush=True)

    if actor_model == '':
        print(f'Did not specify model file. Exiting.', flush=True)
        sys.exit(0)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    policy = Network(num_states, num_actions)

    policy.load_state_dict(torch.load(actor_model))


    # # To evaluate once
    _, _ = eval.rollout(policy, env, render=True)

    # returns = []

    # for i in range(100):
    #     # ep_ret = eval.eval_policy(policy=policy, env=env, render=False)
    #     ep_len, ep_ret = eval.rollout(policy=policy, env=env, render=False)
    #     print(ep_ret)
    #     returns.append(ep_ret/400)
    # plot.plot_testing(returns, 'Figures/eval_returns2.png')


    # success_count = 0
    # for i in range(len(returns)):
    #     if returns[i] >= 100:
    #         success_count += 1

    # print(f'Number of Successful Runs {success_count}')
    # print(f'Number of Failures {100-success_count}')
            


def plotting():
    plot.plot_returns('Data/training_rets4.csv', 'Figures/learning_curve4.png')
    # plot.plot_loss('Data/final_actor_loss_data.csv', 'Figures/final_actor_loss.png', 'Actor Loss', 'olive')
    # plot.plot_loss('Data/final_critic_loss.csv', 'Figures/final_critic_loss.png', 'Critc Loss', 'cyan')


def main(args):

    hyperparameters = {
        'timesteps_per_batch': 2048, 
        'max_timesteps_per_episode': 200, 
        'gamma': 0.99, 
        'n_updates_per_iteration': 10,
        'lr': 3e-4, 
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }
    

    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    elif args.mode == 'test':
        test(env=env, actor_model=args.actor_model)
    elif args.mode == 'plot':
        plotting()
    

if __name__ == '__main__':
    args = get_args()
    main(args)



    
