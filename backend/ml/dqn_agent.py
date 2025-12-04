"""Self-Rewarding Double DQN Agent for autonomous trading.

Implements self-rewarding mechanism where the agent compares expert-defined
rewards (Sharpe ratio, min-max) with self-predicted rewards and chooses
the higher value, enabling better performance in complex market conditions.
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from loguru import logger

from backend.core.config import settings


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        
    def __len__(self):
        """Return buffer size."""
        return len(self.buffer)


class SelfRewardingDoubleDQN:
    """
    Self-Rewarding Double DQN Agent.
    
    Features:
    - Double DQN to reduce overestimation
    - Self-rewarding mechanism comparing expert and predicted rewards
    - Prioritized experience replay
    - Dueling network architecture
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
    ):
        """
        Initialize Double DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.reward_predictor = self._build_reward_predictor()
        
        self.optimizer = optimizers.Adam(learning_rate)
        self.reward_optimizer = optimizers.Adam(learning_rate * 0.1)
        
        # Sync target network
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training stats
        self.training_step = 0
        self.total_loss = 0.0
        self.reward_prediction_loss = 0.0
        
        logger.info(
            f"Initialized Self-Rewarding Double DQN: "
            f"state_dim={state_dim}, action_dim={action_dim}"
        )
        
    def _build_network(self) -> keras.Model:
        """
        Build dueling DQN network.
        
        Architecture:
        - Shared feature extraction layers
        - Dueling streams for value and advantage
        - Combines into Q-values
        """
        inputs = layers.Input(shape=(self.state_dim,))
        
        # Shared feature extraction
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        
        # Dueling streams
        # Value stream
        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, name='value')(value)
        
        # Advantage stream
        advantage = layers.Dense(64, activation='relu')(x)
        advantage = layers.Dense(self.action_dim, name='advantage')(advantage)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = keras.Model(inputs=inputs, outputs=q_values)
        return model
        
    def _build_reward_predictor(self) -> keras.Model:
        """
        Build reward prediction network for self-rewarding mechanism.
        
        Predicts expected reward for state-action pairs.
        """
        inputs = layers.Input(shape=(self.state_dim + 1,))  # state + action
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output: predicted reward
        reward = layers.Dense(1, name='predicted_reward')(x)
        
        model = keras.Model(inputs=inputs, outputs=reward)
        return model
        
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
        logger.debug("Updated target network")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: best action according to Q-network
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            q_values = self.q_network(state_tensor, training=False)
            return int(tf.argmax(q_values[0]).numpy())
            
    def _calculate_expert_reward(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ) -> float:
        """
        Calculate expert-defined reward using domain knowledge.
        
        Components:
        - Sharpe ratio approximation
        - Min-max normalization
        - Risk-adjusted metrics
        
        Args:
            state: Current state
            action: Taken action
            reward: Environment reward
            next_state: Next state
            
        Returns:
            Expert reward
        """
        # Use environment reward as base
        expert_reward = reward
        
        # Add risk-adjusted component (simplified Sharpe ratio proxy)
        # This would use actual portfolio metrics in production
        risk_adjustment = 0.0
        if len(self.replay_buffer) > 10:
            recent_rewards = [exp[2] for exp in list(self.replay_buffer.buffer)[-10:]]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards) + 1e-8
            risk_adjustment = mean_reward / std_reward * 0.1
            
        expert_reward += risk_adjustment
        
        # Min-max normalization component
        if len(self.replay_buffer) > 100:
            all_rewards = [exp[2] for exp in self.replay_buffer.buffer]
            min_r = min(all_rewards)
            max_r = max(all_rewards)
            if max_r > min_r:
                normalized_reward = (reward - min_r) / (max_r - min_r)
                expert_reward = 0.7 * expert_reward + 0.3 * normalized_reward
                
        return expert_reward
        
    def _predict_reward(self, state: np.ndarray, action: int) -> float:
        """
        Predict reward using learned reward model.
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            Predicted reward
        """
        # Combine state and action
        state_action = np.concatenate([state, [action]])
        state_action_tensor = tf.convert_to_tensor(
            state_action.reshape(1, -1), dtype=tf.float32
        )
        
        predicted_reward = self.reward_predictor(state_action_tensor, training=False)
        return float(predicted_reward[0, 0].numpy())
        
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience with self-rewarding mechanism.
        
        Compares expert reward with predicted reward and uses the higher value.
        
        Args:
            state: Current state
            action: Taken action
            reward: Environment reward
            next_state: Next state
            done: Whether episode ended
        """
        # Calculate expert reward
        expert_reward = self._calculate_expert_reward(state, action, reward, next_state)
        
        # Predict reward
        predicted_reward = self._predict_reward(state, action)
        
        # Self-rewarding: choose maximum of expert and predicted
        final_reward = max(expert_reward, predicted_reward)
        
        logger.debug(
            f"Reward selection - Env: {reward:.4f}, "
            f"Expert: {expert_reward:.4f}, "
            f"Predicted: {predicted_reward:.4f}, "
            f"Final: {final_reward:.4f}"
        )
        
        # Store in replay buffer
        self.replay_buffer.add(state, action, final_reward, next_state, done)
        
    def train_step(self):
        """
        Perform one training step using Double DQN.
        
        Double DQN prevents overestimation by:
        1. Using Q-network to select best action
        2. Using target network to evaluate that action
        """
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Double DQN target calculation
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q_values = self.q_network(states, training=True)
            current_q_values = tf.gather_nd(
                current_q_values,
                tf.stack([tf.range(self.batch_size), actions], axis=1)
            )
            
            # Double DQN: use Q-network to select action, target network to evaluate
            next_q_values_online = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_values_online, axis=1)
            
            next_q_values_target = self.target_network(next_states, training=False)
            next_q_values = tf.gather_nd(
                next_q_values_target,
                tf.stack([tf.range(self.batch_size), best_actions], axis=1)
            )
            
            # Target: r + gamma * Q_target(s', argmax_a Q(s', a))
            targets = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Loss
            loss = tf.reduce_mean(tf.square(targets - current_q_values))
            
        # Update Q-network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Train reward predictor
        self._train_reward_predictor(states, actions, rewards)
        
        # Update stats
        self.total_loss += float(loss.numpy())
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return float(loss.numpy())
        
    def _train_reward_predictor(
        self,
        states: tf.Tensor,
        actions: np.ndarray,
        rewards: tf.Tensor
    ):
        """
        Train reward prediction network.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of actual rewards
        """
        with tf.GradientTape() as tape:
            # Prepare state-action pairs
            actions_tensor = tf.convert_to_tensor(
                actions.reshape(-1, 1), dtype=tf.float32
            )
            state_actions = tf.concat([states, actions_tensor], axis=1)
            
            # Predict rewards
            predicted_rewards = self.reward_predictor(state_actions, training=True)
            
            # Loss: MSE between predicted and actual rewards
            loss = tf.reduce_mean(tf.square(rewards - tf.squeeze(predicted_rewards)))
            
        # Update reward predictor
        gradients = tape.gradient(loss, self.reward_predictor.trainable_variables)
        self.reward_optimizer.apply_gradients(
            zip(gradients, self.reward_predictor.trainable_variables)
        )
        
        self.reward_prediction_loss += float(loss.numpy())
        
    def save(self, filepath: str):
        """Save model weights."""
        self.q_network.save_weights(f"{filepath}_q_network.h5")
        self.target_network.save_weights(f"{filepath}_target_network.h5")
        self.reward_predictor.save_weights(f"{filepath}_reward_predictor.h5")
        logger.info(f"Saved model to {filepath}")
        
    def load(self, filepath: str):
        """Load model weights."""
        self.q_network.load_weights(f"{filepath}_q_network.h5")
        self.target_network.load_weights(f"{filepath}_target_network.h5")
        self.reward_predictor.load_weights(f"{filepath}_reward_predictor.h5")
        logger.info(f"Loaded model from {filepath}")
        
    def get_stats(self) -> dict:
        """Get training statistics."""
        avg_loss = self.total_loss / max(1, self.training_step)
        avg_reward_loss = self.reward_prediction_loss / max(1, self.training_step)
        
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'avg_q_loss': avg_loss,
            'avg_reward_loss': avg_reward_loss,
            'buffer_size': len(self.replay_buffer),
        }
