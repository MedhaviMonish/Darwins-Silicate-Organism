import gymnasium as gym
from ppo_model import PPO

env = gym.make("CartPole-v1")
ppo_trainer = PPO(env.observation_space.shape, env.action_space.n , 0, env)
ppo_trainer.learn()
env.close()