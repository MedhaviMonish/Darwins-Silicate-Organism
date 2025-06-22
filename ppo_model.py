import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
from utils import ActionType


class PPOModel(Model):
    def __init__(
        self, input_dim: int, action_dim: int, action_type: ActionType
    ) -> None:
        super(PPOModel, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.action_type: ActionType = action_type
        self.create_models()

    def create_models(self) -> None:
        self.base_model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.input_dim),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.value_model = tf.keras.layers.Dense(1, activation="linear")
        if self.action_type == ActionType.DISCRETE:
            self.action_model = tf.keras.layers.Dense(
                self.action_dim, activation="linear"
            )

    def call(self, observation_logits: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        base_logits = self.base_model(observation_logits)
        value_logits = self.value_model(base_logits)
        if self.action_type == ActionType.DISCRETE:
            action_logits = self.action_model(base_logits)
            log_probs = tf.nn.log_softmax(action_logits)
        else:
            raise NotImplementedError("Only DISCRETE action space is implemented.")

        return log_probs, value_logits

    def sample_action(
        self, observation_logits: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        base_logits = self.base_model(observation_logits)
        value_logits = self.value_model(base_logits)

        if self.action_type == ActionType.DISCRETE:
            action_logits = self.action_model(base_logits)
            action_probs = tf.nn.softmax(action_logits)
            log_probs = tf.math.log(action_probs + 1e-8)
            action = tf.random.categorical(action_logits, num_samples=1)
        else:
            raise NotImplementedError("Only DISCRETE action space is implemented.")

        return tf.squeeze(action, axis=-1), action_probs, log_probs, value_logits
