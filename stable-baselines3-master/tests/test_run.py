import numpy as np
import pytest
import torch
import gym
import matplotlib.pyplot as plt
import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback

os.environ['KMP_DUPLICATE_LIB_OK']='True'


normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func



@pytest.mark.parametrize("model_class", [TD3, DDPG])
@pytest.mark.parametrize("action_noise", [normal_action_noise, OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1 * np.ones(1))])
def test_deterministic_pg(model_class, action_noise):
    """
    Test for DDPG and variants (TD3).
    """
    model = model_class(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        create_eval_env=True,
        buffer_size=250,
        action_noise=action_noise,
    )
    model.learn(total_timesteps=300, eval_freq=250)


@pytest.mark.parametrize("env_id", ["Pendulum-v0"])
def test_a2c(env_id):
    model = A2C("MlpPolicy", env_id,
                seed=0,
                gamma = 0.98,
                normalize_advantage = True,
                max_grad_norm=1,
                use_rms_prop=True,
                gae_lambda = 0.9,
                n_steps = 1,
                learning_rate = 0.00033449110737887957,
                ent_coef = 0.03826151159203985,
                vf_coef= 0.862067985941033,
                buffer_size = 10000,
                batch_size = 512,
                learning_starts = 3000,
                policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=torch.nn.ReLU, ortho_init = False),
                verbose=1,
                create_eval_env=True)
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=250,
                             deterministic=False, render=False)
    model.learn(total_timesteps=50000, eval_freq=100)
    

def test_a2c_2(env_id):
    for buffer_size in [2000, 50000]:
        model = A2C("MlpPolicy", env_id, seed=0, buffer_size = buffer_size, batch_size = 64, learning_starts = 5000, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
        eval_env = gym.make(env_id)
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                     log_path='./logs/a2c_rp', eval_freq=100,
                                     deterministic=False, render=False, n_eval_episodes = 100)
        model.learn(total_timesteps=50000, callback= eval_callback)
        


@pytest.mark.parametrize("env_id", ["Pendulum-v0"])
@pytest.mark.parametrize("clip_range_vf", [None, 0.2, -0.2])
def test_ppo(env_id, clip_range_vf):
    if clip_range_vf is not None and clip_range_vf < 0:
        # Should throw an error
        with pytest.raises(AssertionError):
            model = PPO(
                "MlpPolicy",
                env_id,
                seed=0,
                policy_kwargs=dict(net_arch=[16]),
                verbose=1,
                create_eval_env=True,
                clip_range_vf=clip_range_vf,
            )
    else:
        model = PPO(
            "MlpPolicy",
            env_id,
            n_steps=512,
            seed=0,
            policy_kwargs=dict(net_arch=[16]),
            verbose=1,
            create_eval_env=True,
            clip_range_vf=clip_range_vf,
        )
        eval_env = gym.make(env_id)
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=250,
                             deterministic=True, render=False)
        model.learn(total_timesteps=50000, callback = eval_callback)


@pytest.mark.parametrize("ent_coef", ["auto", 0.2, "auto_0.2"])
def test_sac(ent_coef, i):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=3000,
        verbose=1,
        create_eval_env=True,
        buffer_size=10000,
        ent_coef=ent_coef,
        action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),#,
        target_update_interval = 5000,
        #tensorboard_log="./sac_pendulum_tensorboard/"
    )
    env = model.env
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/without_target', eval_freq=250,
                             deterministic=True, render=False)
    model.learn(total_timesteps=20000, eval_freq = 250)
    """
    definition = 200
    portrait = np.zeros((definition, definition))                                                                       
    state_min = env.observation_space.low                                                                               
    state_max = env.observation_space.high
    for index_t, t in enumerate(np.linspace(-np.pi, np.pi , num=definition)):                               
        for index_td, td in enumerate(np.linspace(state_min[2], state_max[2], num=definition)):                               
            state = torch.Tensor([[np.cos(t), np.sin(t), td]])                                                                            
            action = model.policy.forward(state)
            portrait[definition - (1 + index_td), index_t] = model.critic.q1_forward(state, action)
    plt.figure(figsize=(10, 10))                                                                                        
    plt.imshow(portrait, cmap="inferno", extent=[-180, 180, state_min[2], state_max[2]], aspect='auto')
    plt.rc('axes', titlesize=12) 
    plt.xlabel('angle')
    plt.ylabel('velocity')
    plt.colorbar(label="critic value") 
    plt.scatter([0], [0])
    plt.show()
    #policy = model.policy
    #policy.save("Pendulum-v0#test4SAC#custom#None#{}.zip".format(i))
    #saved_policy = MlpPolicy.load("Pendulum-v0#test4SAC#custom#None#{}.zip".format(i))
    #mean_reward, std_reward = evaluate_policy(saved_policy, model.get_env(), n_eval_episodes=10)
    #print(mean_reward, std_reward)"""
    return model.replay_buffer.rewards

@pytest.mark.parametrize("ent_coef", ["auto", 0.2, "auto_0.2"])
def test_sac2():
    reward = []
    for i in [6000, 8000, 10000]:
        model = SAC(
            "MlpPolicy",
            "Pendulum-v0",
            policy_kwargs=dict(net_arch=[64, 64]),
            learning_starts=5000,
            verbose=0,
            create_eval_env=True,
            buffer_size=i,
            ent_coef=0,
            action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
            batch_size = 32
            )
        eval_env = gym.make('Pendulum-v0')
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                     log_path='./logs/alpha4_histogram', eval_freq=250, n_eval_episodes = 5,
                                     deterministic=True, render=False)
        model.learn(total_timesteps=20000, callback = eval_callback)
        reward.append(eval_callback.last_mean_reward)
        hist, bins = np.histogram(model.replay_buffer.rewards, bins = 500)
        x = []
        for h in range(len(hist)):
            for j in range(hist[h]):
                x.append(bins[h]) 
        plt.hist(x, bins = bins)
        plt.xlabel("reward")
        plt.ylabel("population")
        plt.title("last mean reward = {:.2f} +/- {:.2f}, replay size = {}".format(reward[-1], eval_callback.last_std,  i))
        plt.legend()
        plt.show()

    return reward


@pytest.mark.parametrize("ent_coef", ["auto", 0.2, "auto_0.2"])
def test_sac_phase():
    reward = []
    for i in [2000, 4000, 6000, 8000, 10000]:
        model = SAC(
            "MlpPolicy",
            "Pendulum-v0",
            policy_kwargs=dict(net_arch=[64, 64]),
            learning_starts=5000,
            verbose=0,
            create_eval_env=True,
            buffer_size=i,
            ent_coef=0,
            action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
            batch_size = 32
            )
        env = model.env
        eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                     log_path='./logs/alpha5_phase', eval_freq=250, n_eval_episodes = 100,
                                     deterministic=True, render=False)
        model.learn(total_timesteps=20000, callback = eval_callback)
        reward.append(eval_callback.last_mean_reward)
        definition = 200
        portrait = np.zeros((definition, definition))                                                                       
        state_min = env.observation_space.low                                                                               
        state_max = env.observation_space.high
        for index_t, t in enumerate(np.linspace(-np.pi, np.pi , num=definition)):                               
            for index_td, td in enumerate(np.linspace(state_min[2], state_max[2], num=definition)):                               
                state = torch.Tensor([[np.cos(t), np.sin(t), td]])                                                                            
                action = model.policy.forward(state)
                portrait[definition - (1 + index_td), index_t] = model.critic.q1_forward(state, action)
        plt.figure(figsize=(10, 10))                                                                                        
        plt.imshow(portrait, cmap="inferno", extent=[-180, 180, state_min[2], state_max[2]], aspect='auto')
        plt.rc('axes', titlesize=12) 
        plt.xlabel('angle')
        plt.ylabel('velocity')
        plt.title("critic, last mean reward = {:.2f} +/- {:.2f}, replay size = {}".format(reward[-1], eval_callback.last_std,  i))
        plt.colorbar(label="critic value") 
        plt.scatter([0], [0])
        plt.show()
        definition = 200
        portrait = np.zeros((definition, definition))                                                                       
        state_min = env.observation_space.low                                                                               
        state_max = env.observation_space.high
        portrait = np.zeros((definition, definition))
        for index_t, t in enumerate(np.linspace(-np.pi, np.pi , num=definition)):                               
            for index_td, td in enumerate(np.linspace(state_min[2], state_max[2], num=definition)):
                state = torch.Tensor([[np.cos(t), np.sin(t), td]])
                probs = model.policy.forward(state)
                action = probs.data.numpy().astype(float)
                portrait[definition - (1 + index_td), index_t] = action
        plt.figure(figsize=(10, 10))                                                             
        plt.imshow(portrait, cmap="coolwarm", extent=[-180, 180, state_min[2], state_max[2]], aspect='auto')
        plt.title("action, last mean reward = {:.2f} +/- {:.2f}, replay size = {}".format(reward[-1], eval_callback.last_std,  i))
        plt.colorbar(label="action") 
        plt.rc('axes', titlesize=12) 
        plt.xlabel('angle')
        plt.ylabel('velocity')
        plt.scatter([0], [0]) 
        plt.show()

    return reward


def test_sacs2():
    for i in range(2):
        reward = test_sac2()
        
    return reward
        
    
def param_buff():
    res1 = [0, 0.5, 1, 1.5, 2]
    res2 = [1000, 2000, 10000, 100000, 10 ** 6]
    res3 = [500, 500, 5000, 5000, 5000]
    m = [32, 64]
    
    for o in m: 
        for j, k in zip(res2, res3):
            for i in res1:
                model = SAC(
                    "MlpPolicy",
                    "Pendulum-v0",
                    policy_kwargs=dict(net_arch=[64, 64]),
                    learning_starts=k,
                    verbose=1,
                    create_eval_env=True,
                    buffer_size= j,
                    ent_coef= i,
                    action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
                    batch_size = o
                    #,
                    #tensorboard_log="./sac_pendulum_tensorboard/"
                    )
                eval_env = gym.make('Pendulum-v0')
                eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                             log_path='./logs/alpha3', eval_freq=250,
                                             deterministic=True, render=False)
                model.learn(total_timesteps=20000, callback=eval_callback)
        
    return res1, res2
    
def get_perf(i):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=5e3,
        verbose=1,
        create_eval_env=True,
        buffer_size=1000000,
        ent_coef=0.2,
        action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
        seed = 42
    )
    saved_policy = MlpPolicy.load("Pendulum-v0#test4SAC#custom#None#{}.zip".format(i))
    mean_reward, std_reward = evaluate_policy(saved_policy, model.get_env(), n_eval_episodes=900)
    return mean_reward, std_reward
    

@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test SAC with different number of critics, for TD3, n_critics=1 corresponds to DDPG
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64], n_critics=n_critics),
        learning_starts=100,
        buffer_size=10000,
        verbose=1,
    )
    model.learn(total_timesteps=300)


def test_dqn():
    model = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
        create_eval_env=True,
    )
    model.learn(total_timesteps=500, eval_freq=250)
