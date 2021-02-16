# Basic-Policy-Gradient-Labs

A repo to study basic Policy Gradient algorithms (like REINFORCE) on classic control gym environments

## Example of command

```
python main_pg.py --env_name Pendulum-v0 --nb_repet 1 --nb_cycles 500 --max_episode_steps 200 --policy_type squashedGaussian
```

# SAC 

The SAC version used for our test comes from cleanrl (https://github.com/vwxyzjn/cleanrl) and the results concern about Pendulum.

# SAC of SB3

github of SB3 : https://github.com/DLR-RM/stable-baselines3

to compute the result of SAC, calls on the console test_sac(0.2) of sac.py

Some different results are in the file sac_pendulum_tensorboard : \n
SAC_1 : alpha = 0.2 buffer_size = 100000 \n
SAC_2 : alpha = 0.2 buffer_size = 2000 \n
SAC_3 : alpha = 0.2 buffer_size = 250 \n
SAC_4 : alpha = 0.02 buffer_size = 2000 \n
SAC_5 : alpha = 0.9 buffer_size = 2000 \n
SAC_6 : alpha = 0  buffer_size = 2000 \n 
SAC_7 : alpha = 0.2  buffer_size = 2000 without random samples from replay buffer (instead contiguous samples) \n
