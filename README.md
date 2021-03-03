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

Some different results are in the file sac_pendulum_tensorboard :  
SAC_1 : alpha = 0.2 buffer_size = 100000   
SAC_2 : alpha = 0.2 buffer_size = 2000   
SAC_3 : alpha = 0.2 buffer_size = 250   
SAC_4 : alpha = 0.02 buffer_size = 2000    
SAC_5 : alpha = 0.9 buffer_size = 2000  
SAC_6 : alpha = 0  buffer_size = 2000  
SAC_7 : alpha = 0.2  buffer_size = 2000 without random samples from replay buffer (instead contiguous samples)  

# Results

learning_rate = 0.0007
learning_starts = 5000
soft update coefficient = 0.005
discount factor = 0.99
Minibatch size for each gradient update = 256

sac_pendulum_tensorboard_buff : from SAC_1 to SAC_10 : buffer_sizes : [200, 400, ..., 2000] with 0.2 for alpha  
sac_pendulum_tensorboard_alpha : from SAC_1 to SAC_11 : alphas : [0, 0.02, ..., 0.2] with buffer size 10^6  
sac_pendulum_tensorboard_alpha   from SAC_12 to SAC_22 : alphas : [0, 0.02, ..., 0.2] with buffer size 2000 
