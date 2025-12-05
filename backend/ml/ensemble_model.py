"""Ensemble Model combining multiple ML approaches.

Combines predictions from:
- DQN (Reinforcement Learning)
- LSTM (Deep Learning)
- Statistical Models
"""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger


class EnsembleModel:
    """
    Ensemble model that combines multiple predictors.
    
    Uses weighted voting for final trading decisions.
    """
    
    def __init__(self):
        """Initialize ensemble model."""
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        
        logger.info("Ensemble Model initialized")
        
    def add_model(
        self,
        name: str,
        model: any,
        weight: float = 1.0
    ):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
            weight: Weight in ensemble (will be normalized)
        """
        self.models[name] = model
        self.weights[name] = weight
        self.performance_history[name] = []
        
        logger.info(f"Model added to ensemble: {name} (weight={weight})")
        
    def predict(
        self,
        inputs: Dict[str, any],
        method: str = 'weighted_average'
    ) -> Dict:
        """
        Get ensemble prediction.
        
        Args:
            inputs: Dict mapping model names to their inputs
            method: Aggregation method ('weighted_average', 'voting', 'max_confidence')
            
        Returns:
            Ensemble prediction
        """
        predictions = {}
        confidences = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            if name in inputs:
                try:
                    pred = model.predict(inputs[name])
                    predictions[name] = pred
                    
                    # Extract confidence if available
                    if isinstance(pred, dict) and 'confidence' in pred:
                        confidences[name] = pred['confidence']
                    else:
                        confidences[name] = 1.0
                except Exception as e:
                    logger.error(f"Error getting prediction from {name}: {e}")
                    continue
                    
        if not predictions:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No valid predictions'}
            
        # Aggregate predictions
        if method == 'weighted_average':
            return self._weighted_average(predictions, confidences)
        elif method == 'voting':
            return self._voting(predictions, confidences)
        elif method == 'max_confidence':
            return self._max_confidence(predictions, confidences)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    def _weighted_average(
        self,
        predictions: Dict,
        confidences: Dict
    ) -> Dict:
        """
        Weighted average aggregation.
        
        Combines model predictions weighted by their assigned weights
        and confidence scores.
        """
        # Normalize weights
        total_weight = sum(self.weights.values())
        normalized_weights = {
            k: v / total_weight for k, v in self.weights.items()
        }
        
        # For classification (buy/sell/hold), use weighted voting
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        
        for name, pred in predictions.items():
            if name not in normalized_weights:
                continue
                
            action = pred if isinstance(pred, str) else pred.get('action', 'hold')
            weight = normalized_weights[name] * confidences.get(name, 1.0)
            
            if action in votes:
                votes[action] += weight
                
        # Get action with highest weight
        best_action = max(votes.items(), key=lambda x: x[1])
        
        return {
            'action': best_action[0],
            'confidence': best_action[1],
            'method': 'weighted_average',
            'votes': votes
        }
        
    def _voting(self, predictions: Dict, confidences: Dict) -> Dict:
        """Simple majority voting."""
        votes = {}
        
        for pred in predictions.values():
            action = pred if isinstance(pred, str) else pred.get('action', 'hold')
            votes[action] = votes.get(action, 0) + 1
            
        best_action = max(votes.items(), key=lambda x: x[1])
        
        return {
            'action': best_action[0],
            'confidence': best_action[1] / len(predictions),
            'method': 'voting',
            'votes': votes
        }
        
    def _max_confidence(self, predictions: Dict, confidences: Dict) -> Dict:
        """Select prediction with highest confidence."""
        if not confidences:
            return self._voting(predictions, confidences)
            
        max_conf_model = max(confidences.items(), key=lambda x: x[1])
        pred = predictions[max_conf_model[0]]
        
        return {
            'action': pred if isinstance(pred, str) else pred.get('action', 'hold'),
            'confidence': max_conf_model[1],
            'method': 'max_confidence',
            'selected_model': max_conf_model[0]
        }
        
    def update_performance(
        self,
        model_name: str,
        performance: float
    ):
        """
        Update model performance history.
        
        Used for adaptive weighting.
        
        Args:
            model_name: Name of model
            performance: Performance metric (e.g., return, accuracy)
        """
        if model_name in self.performance_history:
            self.performance_history[model_name].append(performance)
            
            # Keep only recent history
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name] = \
                    self.performance_history[model_name][-100:]
                    
    def adaptive_weighting(self, lookback: int = 20):
        """
        Adaptively adjust weights based on recent performance.
        
        Args:
            lookback: Number of recent results to consider
        """
        for name in self.models.keys():
            if name not in self.performance_history:
                continue
                
            recent_perf = self.performance_history[name][-lookback:]
            
            if len(recent_perf) >= 5:
                # Use average recent performance as weight
                avg_perf = np.mean(recent_perf)
                
                # Convert to positive weight (shift if necessary)
                min_perf = min([np.mean(h[-lookback:]) 
                               for h in self.performance_history.values() 
                               if len(h) >= 5])
                               
                if min_perf < 0:
                    adj_weight = avg_perf - min_perf + 0.1
                else:
                    adj_weight = max(avg_perf, 0.1)
                    
                self.weights[name] = adj_weight
                
        logger.info(f"Weights updated adaptively: {self.weights}")
        
    def get_model_stats(self) -> Dict:
        """Get statistics for all models."""
        stats = {}
        
        for name in self.models.keys():
            if name in self.performance_history:
                history = self.performance_history[name]
                if len(history) > 0:
                    stats[name] = {
                        'weight': self.weights.get(name, 1.0),
                        'avg_performance': np.mean(history),
                        'recent_performance': np.mean(history[-20:]) if len(history) >= 20 else np.mean(history),
                        'samples': len(history)
                    }
                    
        return stats


# Global instance
ensemble_model = EnsembleModel()
