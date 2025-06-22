import tensorflow as tf
from ppo_model import PPOModel
from buffer import RolloutBuffer
from gymnasium import Env
import numpy as np
from utils import ActionType
from advantage import get_gaes


class PPOTrainer:
    def __init__(self, input_dim, action_dim, action_type: ActionType, env: Env):
        self.gamma = 0.95
        self.lamda = 0.9
        self.ppo_eps = 0.2
        self.normalize = True
        self.epoch = 3
        self.rollout = 256
        self.batch_size = 128

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_type: ActionType = action_type
        self.env: Env = env
        self.model = PPOModel(input_dim, action_dim, action_type)
        self.buffer = RolloutBuffer()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.0003)

    def run_episode(self):
        total_timesteps = 0
        while total_timesteps < self.rollout:
            obs, _ = self.env.reset()  # No fixed seed for diversity
            done = False
            r = 0

            while not done:
                obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
                action, logits, value = self.model.sample_action(obs_tensor)
                action = action.numpy()

                next_obs, reward, terminated, truncated, _ = self.env.step(action[0])
                done_flag = int(terminated or truncated)

                # Store transition
                self.buffer.add(obs, action, reward, done_flag, next_obs, logits, value)

                obs = next_obs
                total_timesteps += 1

                r += reward
                if done_flag:
                    print(f"Reward before end {r}")

                if done_flag or total_timesteps >= self.rollout:
                    break

    def update_model(self):
        records = self.buffer.get()

        # Convert buffer data to tensors
        observations = tf.convert_to_tensor(records["states"], dtype=tf.float32)
        next_observations = tf.convert_to_tensor(
            records["next_states"], dtype=tf.float32
        )
        rewards = tf.convert_to_tensor(records["rewards"], dtype=tf.float32)
        dones = tf.convert_to_tensor(records["dones"], dtype=tf.float32)
        actions = tf.convert_to_tensor(records["actions"], dtype=tf.int32)
        old_policies = tf.convert_to_tensor(records["log_probs"], dtype=tf.float32)
        current_values = tf.convert_to_tensor(records["values"], dtype=tf.float32)

        # Get value predictions for next states
        _, next_values = self.model(next_observations)
        current_values = tf.squeeze(current_values, axis=-1)
        next_values = tf.squeeze(next_values, axis=-1)

        # Compute advantage and target returns
        advantages, targets = self.compute_gae(
            rewards, dones, current_values, next_values
        )

        for _ in range(self.epoch):
            indices = np.arange(self.rollout)
            np.random.shuffle(indices)
            sample_idx = indices[: self.batch_size]

            batch_states = tf.gather(observations, sample_idx)
            batch_actions = tf.gather(actions, sample_idx)
            batch_advantages = tf.gather(advantages, sample_idx)
            batch_targets = tf.gather(targets, sample_idx)
            batch_old_policies = tf.gather(old_policies, sample_idx)
            print(batch_advantages)
            print(batch_targets)

            with tf.GradientTape() as tape:
                new_policies, new_values = self.model(batch_states)
                new_values = tf.squeeze(new_values, axis=-1)

                # Entropy for exploration encouragement
                entropy = (
                    tf.reduce_mean(-new_policies * tf.math.log(new_policies + 1e-8))
                    * 0.1
                )

                # Get selected action probabilities
                onehot_actions = tf.one_hot(batch_actions, self.action_dim)
                selected_probs = tf.reduce_sum(new_policies * onehot_actions, axis=1)
                selected_old_probs = tf.reduce_sum(
                    batch_old_policies * onehot_actions, axis=1
                )

                # PPO clipped surrogate loss
                ratio = selected_probs / (selected_old_probs + 1e-8)
                clipped_ratio = tf.clip_by_value(
                    ratio, 1 - self.ppo_eps, 1 + self.ppo_eps
                )
                surrogate = tf.minimum(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                )
                policy_loss = -tf.reduce_mean(surrogate) + entropy

                # Value loss
                value_loss = tf.reduce_mean(tf.square(batch_targets - new_values))

                # Total loss
                total_loss = policy_loss + value_loss

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def compute_gae(self, rewards, dones, values, next_values):
        return get_gaes(
            rewards, dones, values, next_values, self.gamma, self.lamda, self.normalize
        )

    def train(self, total_episodes=1000):
        running_reward = 0
        for episode in range(total_episodes):
            self.buffer.reset()
            self.run_episode()
            self.update_model()

            rewards = np.array(self.buffer.rewards)
            dones = np.array(self.buffer.dones)

            episode_count = np.sum(dones)  # how many full env runs
            total_reward = np.sum(rewards)

            avg_episode_reward = total_reward / (max(episode_count, 1))
            smoothed_reward = (
                0.05 * avg_episode_reward + 0.95 * smoothed_reward
                if episode > 0
                else avg_episode_reward
            )

            print(
                f"Episode {episode} | Avg Episode Reward: {avg_episode_reward:.2f} | Smoothed Avg: {smoothed_reward:.2f} | Episode Count: {int(episode_count)}"
            )
