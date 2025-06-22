import tensorflow as tf
from ppo_model import PPOModel
from buffer import RolloutBuffer
from logger import MetricsLogger
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
        self.logger = MetricsLogger()
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

                if done_flag or total_timesteps >= self.rollout:
                    break

    def update_model(self):
        records = self.buffer.get()

        # Convert buffer data to tensors
        states = tf.convert_to_tensor(records["states"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(records["next_states"], dtype=tf.float32)
        rewards = tf.convert_to_tensor(records["rewards"], dtype=tf.float32)
        dones = tf.convert_to_tensor(records["dones"], dtype=tf.float32)
        actions = tf.convert_to_tensor(records["actions"], dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(records["log_probs"], dtype=tf.float32)
        values = tf.convert_to_tensor(records["values"], dtype=tf.float32)

        # Get next state value estimates
        _, next_values = self.model(next_states)
        values = tf.squeeze(values, axis=-1)
        next_values = tf.squeeze(next_values, axis=-1)

        # Compute advantage and TD targets
        advantages, targets = self.compute_gae(rewards, dones, values, next_values)

        # Training loop
        for _ in range(self.epoch):
            indices = np.arange(self.rollout)
            np.random.shuffle(indices)
            batch_idx = indices[: self.batch_size]

            batch_states = tf.gather(states, batch_idx)
            batch_actions = tf.gather(actions, batch_idx)
            batch_advantages = tf.gather(advantages, batch_idx)
            batch_targets = tf.gather(targets, batch_idx)
            batch_old_log_probs = tf.gather(old_log_probs, batch_idx)

            with tf.GradientTape() as tape:
                policy_logits, predicted_values = self.model(batch_states)
                predicted_values = tf.squeeze(predicted_values, axis=-1)

                # Entropy bonus for exploration
                entropy = (
                    tf.reduce_mean(-policy_logits * tf.math.log(policy_logits + 1e-8))
                    * 0.1
                )

                # Log-probs for current policy
                onehot = tf.one_hot(batch_actions, self.action_dim)
                selected_probs = tf.reduce_sum(policy_logits * onehot, axis=1)
                selected_old_probs = tf.reduce_sum(batch_old_log_probs * onehot, axis=1)

                log_pi = tf.math.log(selected_probs + 1e-8)
                log_old_pi = tf.math.log(selected_old_probs + 1e-8)

                # PPO clipped loss
                ratio = tf.exp(log_pi - log_old_pi)
                clipped = tf.clip_by_value(ratio, 1 - self.ppo_eps, 1 + self.ppo_eps)
                surrogate = tf.minimum(
                    ratio * batch_advantages, clipped * batch_advantages
                )
                policy_loss = -tf.reduce_mean(surrogate) + entropy

                # Value function loss
                value_loss = tf.reduce_mean(tf.square(batch_targets - predicted_values))

                # Total loss
                total_loss = policy_loss + value_loss

            # Apply gradients
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            self.logger.log_scalar("value_loss", value_loss.numpy())
            self.logger.log_scalar("policy_loss", policy_loss.numpy())
            self.logger.log_scalar("entropy", entropy.numpy())
            self.logger.log_scalar("adv_mean", tf.reduce_mean(batch_advantages).numpy())
            self.logger.log_scalar(
                "adv_std", tf.math.reduce_std(batch_advantages).numpy()
            )
            self.logger.log_scalar("log_prob_std", tf.math.reduce_std(log_pi).numpy())

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

            episode_reward = np.sum(self.buffer.rewards)
            num_episodes = np.sum(self.buffer.dones)
            avg_episode_reward = episode_reward / (max(num_episodes, 1))

            running_reward = 0.05 * avg_episode_reward + (1 - 0.05) * running_reward

            print(
                f"Episode {episode} | Avg Reward: {avg_episode_reward:.2f} | Running Avg: {running_reward:.2f}"
            )
            self.logger.log_scalar("episode_reward", episode_reward)
            self.logger.log_scalar("avg_episode_reward", avg_episode_reward)
            self.logger.log_scalar("running_reward", running_reward)
            self.logger.write()
