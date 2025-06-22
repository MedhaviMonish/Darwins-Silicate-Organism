from ppo_trainer import PPOTrainer
from utils import ActionType
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape
action_dim = env.action_space.n

agent = PPOTrainer(obs_dim, action_dim, ActionType.DISCRETE, env)
agent.train(total_episodes=1000)
