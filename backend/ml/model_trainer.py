"""Model trainer with continuous learning and hyperparameter optimization."""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from backend.core.config import settings
from backend.core.throttler import throttler
from backend.ml.rl_environment import TradingEnvironment
from backend.ml.dqn_agent import SelfRewardingDoubleDQN
from backend.data.polygon_client import polygon_client


class ModelTrainer:
    """
    Automated model trainer with continuous learning.
    
    Features:
    - Hyperparameter optimization
    - Continuous learning from new data
    - Performance monitoring
    - Automatic retraining triggers
    """
    
    def __init__(
        self,
        model_save_path: str = "models/",
        performance_threshold: float = 0.05,  # Min Sharpe ratio to keep model
        retrain_interval_hours: int = 24,
    ):
        """
        Initialize model trainer.
        
        Args:
            model_save_path: Directory to save models
            performance_threshold: Minimum performance to keep model
            retrain_interval_hours: Hours between retraining
        """
        self.model_save_path = model_save_path
        self.performance_threshold = performance_threshold
        self.retrain_interval_hours = retrain_interval_hours
        
        # Ensure model directory exists
        os.makedirs(model_save_path, exist_ok=True)
        
        # Training history
        self.training_history = []
        self.best_performance = -np.inf
        self.last_training_time = None
        
        logger.info("Model trainer initialized")
        
    async def train_model(
        self,
        symbol: str,
        episodes: int = 100,
        hyperparams: Optional[Dict] = None,
        save_best: bool = True,
    ) -> Tuple[SelfRewardingDoubleDQN, Dict]:
        """
        Train a new model.
        
        Args:
            symbol: Stock symbol to train on
            episodes: Number of training episodes
            hyperparams: Hyperparameters for agent
            save_best: Whether to save best model
            
        Returns:
            Trained agent and performance metrics
        """
        logger.info(f"Starting training for {symbol} - {episodes} episodes")
        
        # Fetch historical data
        await throttler.throttle_if_needed()
        historical_data = await polygon_client.get_historical_bars(
            symbol,
            start_date=datetime.now() - pd.Timedelta(days=730),  # 2 years
            end_date=datetime.now(),
            timespan="day"
        )
        
        if not historical_data:
            logger.error(f"No historical data for {symbol}")
            return None, {}
            
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df.set_index('timestamp', inplace=True)
        
        # Add RSI indicator for environment
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Create environment
        env = TradingEnvironment(
            data=df,
            initial_balance=settings.backtest_initial_capital
        )
        
        # Default hyperparameters
        default_hyperparams = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'target_update_freq': 1000,
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
            
        # Create agent
        agent = SelfRewardingDoubleDQN(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            **default_hyperparams
        )
        
        # Training loop
        episode_rewards = []
        episode_returns = []
        best_return = -np.inf
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Select action
                action = agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience (with self-rewarding)
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                if step % 4 == 0:  # Train every 4 steps
                    loss = agent.train_step()
                    
                episode_reward += reward
                state = next_state
                step += 1
                
                # CPU throttling
                if step % 100 == 0:
                    await throttler.throttle_if_needed()
                    
            # Episode completed
            metrics = env.get_performance_metrics()
            episode_return = metrics['total_return']
            episode_rewards.append(episode_reward)
            episode_returns.append(episode_return)
            
            # Save best model
            if episode_return > best_return and save_best:
                best_return = episode_return
                model_path = os.path.join(
                    self.model_save_path,
                    f"{symbol}_best"
                )
                agent.save(model_path)
                logger.info(
                    f"New best model saved - Return: {episode_return*100:.2f}%"
                )
                
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_return = np.mean(episode_returns[-10:])
                stats = agent.get_stats()
                
                logger.info(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Avg Return: {avg_return*100:.2f}% | "
                    f"Epsilon: {stats['epsilon']:.4f} | "
                    f"Q Loss: {stats['avg_q_loss']:.6f}"
                )
                
        # Final evaluation
        final_metrics = self._evaluate_model(agent, env, episodes=5)
        
        # Store training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'episodes': episodes,
            'hyperparams': default_hyperparams,
            'final_metrics': final_metrics,
            'best_return': best_return,
        }
        self.training_history.append(training_record)
        self._save_training_history()
        
        # Update best performance
        if final_metrics['sharpe_ratio'] > self.best_performance:
            self.best_performance = final_metrics['sharpe_ratio']
            
        self.last_training_time = datetime.now()
        
        logger.info(f"Training completed - Sharpe: {final_metrics['sharpe_ratio']:.2f}")
        
        return agent, final_metrics
        
    def _evaluate_model(
        self,
        agent: SelfRewardingDoubleDQN,
        env: TradingEnvironment,
        episodes: int = 5
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            agent: Trained agent
            env: Trading environment
            episodes: Number of evaluation episodes
            
        Returns:
            Performance metrics
        """
        all_metrics = []
        
        for _ in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
                
            metrics = env.get_performance_metrics()
            all_metrics.append(metrics)
            
        # Average metrics
        avg_metrics = {
            'total_return': np.mean([m['total_return'] for m in all_metrics]),
            'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics]),
            'max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'win_rate': np.mean([m['win_rate'] for m in all_metrics]),
            'total_trades': np.mean([m['total_trades'] for m in all_metrics]),
        }
        
        return avg_metrics
        
    def should_retrain(self, current_performance: float) -> bool:
        """
        Check if model should be retrained.
        
        Args:
            current_performance: Current Sharpe ratio
            
        Returns:
            True if should retrain
        """
        # Performance-based trigger
        if current_performance < self.performance_threshold:
            logger.warning(
                f"Performance below threshold: {current_performance:.4f} < "
                f"{self.performance_threshold:.4f}"
            )
            return True
            
        # Time-based trigger
        if self.last_training_time:
            hours_since_training = (
                datetime.now() - self.last_training_time
            ).total_seconds() / 3600
            
            if hours_since_training >= self.retrain_interval_hours:
                logger.info(
                    f"Retraining due to time trigger: "
                    f"{hours_since_training:.1f} hours since last training"
                )
                return True
                
        return False
        
    async def optimize_hyperparameters(
        self,
        symbol: str,
        n_trials: int = 20,
    ) -> Dict:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            symbol: Stock symbol
            n_trials: Number of trials
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for {symbol}")
        
        # Define search space
        param_grid = {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'gamma': [0.95, 0.99, 0.999],
            'epsilon_decay': [0.99, 0.995, 0.999],
            'batch_size': [32, 64, 128],
        }
        
        best_params = None
        best_sharpe = -np.inf
        
        for trial in range(n_trials):
            # Random sample from grid
            params = {
                key: np.random.choice(values)
                for key, values in param_grid.items()
            }
            
            logger.info(f"Trial {trial + 1}/{n_trials}: {params}")
            
            # Train with these parameters
            agent, metrics = await self.train_model(
                symbol,
                episodes=50,  # Fewer episodes for optimization
                hyperparams=params,
                save_best=False
            )
            
            sharpe = metrics['sharpe_ratio']
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
                logger.info(f"New best params found - Sharpe: {sharpe:.4f}")
                
            # CPU throttling between trials
            await throttler.throttle_if_needed()
            
        logger.info(f"Optimization complete - Best Sharpe: {best_sharpe:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _save_training_history(self):
        """Save training history to file."""
        history_file = os.path.join(self.model_save_path, 'training_history.json')
        
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
        logger.debug(f"Training history saved to {history_file}")
        
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'total_trainings': len(self.training_history),
            'best_performance': self.best_performance,
            'last_training': (
                self.last_training_time.isoformat()
                if self.last_training_time
                else None
            ),
        }


# Global instance
model_trainer = ModelTrainer()
