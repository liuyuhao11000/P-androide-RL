import numpy as np
import pytest
import torch
import gym

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback


normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))


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


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
def test_a2c(env_id):
    model = A2C("MlpPolicy", env_id, seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1, create_eval_env=True)
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
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
        model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("ent_coef", ["auto", 0.2, "auto_0.2"])
def test_sac(ent_coef, i):
    model = SAC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=5e3,
        verbose=1,
        create_eval_env=True,
        buffer_size=1000000,
        ent_coef=ent_coef,
        action_noise=NormalActionNoise(np.zeros(1), np.zeros(1))#,
        #tensorboard_log="./sac_pendulum_tensorboard/"
    )
    eval_env = gym.make('Pendulum-v0')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=250,
                             deterministic=True, render=False)
    model.learn(total_timesteps=20000, callback=eval_callback)
    #policy = model.policy
    #policy.save("Pendulum-v0#test4SAC#custom#None#{}.zip".format(i))
    #saved_policy = MlpPolicy.load("Pendulum-v0#test4SAC#custom#None#{}.zip".format(i))
    #mean_reward, std_reward = evaluate_policy(saved_policy, model.get_env(), n_eval_episodes=10)
    #print(mean_reward, std_reward)
    
def param_buff():
    res1 = [0, 0.5, 1, 1.5, 2]
    res2 = [1000, 2000, 10000, 100000, 10 ** 6]
    res3 = [500, 500, 5000, 5000, 5000]
    
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
                action_noise=NormalActionNoise(np.zeros(1), np.zeros(1))#,
                #tensorboard_log="./sac_pendulum_tensorboard/"
                )
            eval_env = gym.make('Pendulum-v0')
            eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                         log_path='./logs/alpha2c', eval_freq=250,
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
