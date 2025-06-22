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
                action, action_probs, log_probs, value = self.model.sample_action(
                    obs_tensor
                )
                action = action.numpy()

                next_obs, reward, terminated, truncated, _ = self.env.step(action[0])
                done_flag = int(terminated or truncated)

                # Store transition
                self.buffer.add(
                    obs,
                    action,
                    reward,
                    done_flag,
                    next_obs,
                    log_probs[0].numpy(),
                    value.numpy()[0],
                )

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

        # Compute values for next state
        _, next_values = self.model(next_states)
        values = tf.squeeze(values, axis=-1)
        next_values = tf.squeeze(next_values, axis=-1)

        # GAE: compute advantages and TD targets
        advantages, targets = self.compute_gae(rewards, dones, values, next_values)

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
                log_probs, predicted_values = self.model(batch_states)
                predicted_values = tf.squeeze(predicted_values, axis=-1)
                # Entropy bonus (based on log_probs = log_softmax)
                probs = tf.exp(log_probs)
                entropy = (
                    -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1)) * 0.1
                )

                # Flatten actions to shape [128]
                batch_actions_flat = tf.squeeze(
                    batch_actions, axis=-1
                )  # or tf.reshape(..., [-1])

                # Use tf.gather on each row (i.e., gather index from axis=1 per row)
                # Result shape: [128], contains log prob of taken action per sample
                selected_log_probs = tf.reduce_sum(
                    log_probs * tf.one_hot(batch_actions_flat, self.action_dim), axis=1
                )
                selected_old_log_probs = tf.reduce_sum(
                    batch_old_log_probs
                    * tf.one_hot(batch_actions_flat, self.action_dim),
                    axis=1,
                )

                # PPO clipped objective
                ratio = tf.exp(selected_log_probs - selected_old_log_probs)
                clipped_ratio = tf.clip_by_value(
                    ratio, 1 - self.ppo_eps, 1 + self.ppo_eps
                )

                surrogate = tf.minimum(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                )

                policy_loss = -tf.reduce_mean(surrogate) + entropy

                # Value function loss
                value_loss = tf.reduce_mean(tf.square(batch_targets - predicted_values))

                total_loss = policy_loss + value_loss

            # Optimize
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            # ✅ Logging
            self.logger.log_scalar("loss/total", total_loss.numpy())
            self.logger.log_scalar("loss/policy", policy_loss.numpy())
            self.logger.log_scalar("loss/value", value_loss.numpy())
            self.logger.log_scalar("entropy", entropy.numpy())
            self.logger.log_scalar(
                "advantage/mean", tf.reduce_mean(batch_advantages).numpy()
            )
            self.logger.log_scalar(
                "advantage/std", tf.math.reduce_std(batch_advantages).numpy()
            )
            self.logger.log_scalar(
                "log_prob/std", tf.math.reduce_std(selected_log_probs).numpy()
            )

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
