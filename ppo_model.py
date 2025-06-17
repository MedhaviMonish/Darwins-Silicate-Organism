import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from gymnasium import Env
import numpy as np
from utils import ActionType
from advantage import get_gaes

class PPO(Model):
    def __init__ (self, input_dim, action_dim, action_type:ActionType, env:Env):
        super(PPO, self).__init__()
        self.learning_rate = 0.0003
        self.gamma = 0.95
        self.lamda = 0.9
        self.ppo_eps = 0.2
        self.normalize = True
        self.epoch = 3
        self.rollout = 256
        self.batch_size = 128

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_type:ActionType = action_type
        self.env:Env = env
        self.create_models()
        self.opt = optimizers.Adam(learning_rate=self.learning_rate, )
    
    def create_models(self):
        self.base_model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_dim),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')
        ])
        self.value_model = tf.keras.layers.Dense(1, activation='linear')
        if self.action_type == ActionType.DISCRETE:
            self.action_model = tf.keras.layers.Dense(self.action_dim, activation='linear')
    
    def sample_action(self, observation_logits):
        base_logits = self.base_model(observation_logits)
        if self.action_type == ActionType.DISCRETE:
            action_logits = self.action_model(base_logits)
            # Sample from the categorical distribution, this method does softmax behind the scene
            action = tf.random.categorical(action_logits, num_samples=1)
            return tf.squeeze(action, axis=-1)

    def call(self, observation_logits):
        base_logits = self.base_model(observation_logits)
        value_logits = self.value_model(base_logits)
        if self.action_type == ActionType.DISCRETE:
            action_logits = self.action_model(base_logits)
            action_logits = tf.nn.softmax(action_logits)
        
        return action_logits, value_logits
    
    def update_models(self, observation_list, next_observation_list, reward_list, done_list, action_list):
        observations = tf.convert_to_tensor(observation_list)
        next_observations = tf.convert_to_tensor(next_observation_list)
        rewards = tf.convert_to_tensor(reward_list)
        done = tf.convert_to_tensor(done_list, dtype=tf.float32)
        actions = tf.convert_to_tensor(action_list)

        old_policy, current_value = self(observations)
        _, next_value = self(next_observations)
        current_value = tf.squeeze(current_value, axis=-1)
        next_value = tf.squeeze(next_value, axis=-1)

        adv, target = get_gaes(
            rewards=rewards,
            dones=done,
            values=current_value,
            next_values=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize
        )
        # print(f"Total run in this episode {len(observation_list)}")
        # print(self.trainable_variables)
        for _ in range(self.epoch):
            sample_range = np.arange(self.rollout)
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]

            batch_state = tf.gather(observations, sample_idx)
            batch_action = tf.gather(actions, sample_idx)
            batch_target = tf.gather(target, sample_idx)
            batch_adv = tf.gather(adv, sample_idx)
            batch_old_policy = tf.gather(old_policy, sample_idx)

            ppo_variable = self.trainable_variables
            with tf.GradientTape() as tape:
                train_policy, train_current_value = self(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value, axis=-1)
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)

                entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8)) * 0.1
                onehot_action = tf.one_hot(train_action, self.action_dim)
                selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                logpi = tf.math.log(selected_prob + 1e-8)
                logoldpi = tf.math.log(selected_old_prob + 1e-8)

                ratio = tf.exp(logpi - logoldpi)
                clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1-self.ppo_eps, clip_value_max=1+self.ppo_eps)
                minimum = tf.minimum(tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio))
                pi_loss = -tf.reduce_mean(minimum) + entropy

                value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))

                total_loss = pi_loss + value_loss

            grads = tape.gradient(total_loss, ppo_variable)
            self.opt.apply_gradients(zip(grads, ppo_variable))        

    
    def learn(self, episodes=1000):
        recent_rewards = []

        for episode in range(episodes):
            observation_list, next_observation_list = [], []
            reward_list, done_list, action_list = [], [], []
            total_timesteps = 0
            episode_rewards = []

            while total_timesteps < self.rollout:
                obs, _ = self.env.reset()  # No fixed seed for diversity
                done = False
                ep_reward = 0

                while not done:
                    obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
                    action = self.sample_action(obs_tensor)
                    action = action.numpy()

                    next_obs, reward, terminated, truncated, _ = self.env.step(action[0])
                    done_flag = int(terminated or truncated)
                    if terminated:
                        reward = -2.0  # Pole fell
                    elif truncated:
                        reward = 2.0 
                    # Store transition
                    observation_list.append(obs)
                    next_observation_list.append(next_obs)
                    reward_list.append(reward)
                    done_list.append(done_flag)
                    action_list.append(action[0])

                    obs = next_obs
                    ep_reward += reward
                    total_timesteps += 1

                    if done_flag or total_timesteps >= self.rollout:
                        break

                episode_rewards.append(ep_reward)
            # Update PPO model
            self.update_models(observation_list, next_observation_list, reward_list, done_list, action_list)

            # Log reward
            avg_reward = np.mean(episode_rewards)
            recent_rewards.append(avg_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            running_avg = np.mean(recent_rewards)

            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Running Avg: {running_avg:.2f}")
