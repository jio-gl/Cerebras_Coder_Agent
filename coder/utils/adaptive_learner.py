"""
Adaptive Learning Mechanism implementing the learning framework from the paper.

This module implements the adaptive learning mechanism described in the paper:
- Parameter updates: θ_{t+1} = θ_t + η_t ∇_θ J(θ_t)
- Convergence analysis
- Learning rate adaptation
- Performance-based optimization
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math


@dataclass
class LearningParameters:
    """Represents the learning parameters that can be adapted."""
    learning_rate: float = 0.01
    utility_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.3,
        "success_rate": 0.2,
        "efficiency": 0.2,
        "simplicity": 0.15,
        "context": 0.15
    })
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.4,
        "efficiency": 0.3,
        "readability": 0.3
    })
    discount_factor: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "learning_rate": self.learning_rate,
            "utility_weights": self.utility_weights.copy(),
            "quality_weights": self.quality_weights.copy(),
            "discount_factor": self.discount_factor
        }
    
    def update(self, gradients: Dict[str, float], learning_rate: float):
        """
        Update parameters using gradients.
        
        Implements the update rule from the paper: θ_{t+1} = θ_t + η_t ∇_θ J(θ_t)
        """
        # Update utility weights
        for key in self.utility_weights:
            if key in gradients:
                self.utility_weights[key] += learning_rate * gradients[key]
                # Ensure weights stay positive and sum to reasonable values
                self.utility_weights[key] = max(0.0, min(1.0, self.utility_weights[key]))
        
        # Update quality weights
        for key in self.quality_weights:
            if key in gradients:
                self.quality_weights[key] += learning_rate * gradients[key]
                # Ensure weights stay positive and sum to reasonable values
                self.quality_weights[key] = max(0.0, min(1.0, self.quality_weights[key]))
        
        # Normalize weights to sum to 1
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to ensure they sum to 1."""
        # Normalize utility weights
        utility_sum = sum(self.utility_weights.values())
        if utility_sum > 0:
            for key in self.utility_weights:
                self.utility_weights[key] /= utility_sum
        
        # Normalize quality weights
        quality_sum = sum(self.quality_weights.values())
        if quality_sum > 0:
            for key in self.quality_weights:
                self.quality_weights[key] /= quality_sum


@dataclass
class LearningRecord:
    """Represents a learning step record."""
    timestamp: float
    parameters: LearningParameters
    performance_metrics: Dict[str, float]
    gradients: Dict[str, float]
    learning_rate: float
    convergence_metric: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "parameters": self.parameters.to_dict(),
            "performance_metrics": self.performance_metrics,
            "gradients": self.gradients,
            "learning_rate": self.learning_rate,
            "convergence_metric": self.convergence_metric
        }


class AdaptiveLearner:
    """
    Implements the adaptive learning mechanism from the paper.
    
    This class manages the learning process, parameter updates, and convergence analysis
    as described in the theoretical framework.
    """
    
    def __init__(self, 
                 initial_parameters: Optional[LearningParameters] = None,
                 data_file: Optional[str] = None,
                 load_existing_data: bool = True):
        """
        Initialize the adaptive learner.
        
        Args:
            initial_parameters: Initial learning parameters
            data_file: Optional file path for persistent storage
            load_existing_data: Whether to load existing data from file (default: True)
        """
        self.parameters = initial_parameters or LearningParameters()
        self.data_file = data_file or "learning_data.json"
        self.learning_history: List[LearningRecord] = []
        self.convergence_threshold = 0.001
        self.max_iterations = 1000
        
        # Load existing data if available and requested
        if load_existing_data:
            self._load_data()
    
    def compute_gradients(self, 
                         current_performance: Dict[str, float],
                         target_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Compute gradients for parameter updates.
        
        This implements the gradient computation for the learning algorithm.
        """
        gradients = {}
        
        # Compute gradients for utility weights based on performance differences
        for key in self.parameters.utility_weights:
            current_val = current_performance.get(key, 0.0)
            target_val = target_performance.get(key, 0.8)  # Target 80% performance
            gradients[key] = round(target_val - current_val, 10)  # Round to handle floating point precision
        
        # Compute gradients for quality weights
        for key in self.parameters.quality_weights:
            current_val = current_performance.get(f"quality_{key}", 0.0)
            target_val = target_performance.get(f"quality_{key}", 0.8)
            gradients[key] = round(target_val - current_val, 10)  # Round to handle floating point precision
        
        return gradients
    
    def update_parameters(self, 
                         performance_metrics: Dict[str, float],
                         target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update learning parameters based on performance.
        
        Implements the parameter update rule from the paper.
        """
        if target_metrics is None:
            target_metrics = {
                "success_rate": 0.9,
                "average_quality_score": 0.8,
                "efficiency": 0.8
            }
        
        # Compute gradients
        gradients = self.compute_gradients(performance_metrics, target_metrics)
        
        # Adaptive learning rate based on gradient magnitude
        gradient_magnitude = math.sqrt(sum(g**2 for g in gradients.values()))
        adaptive_learning_rate = self.parameters.learning_rate / (1.0 + gradient_magnitude)
        
        # Update parameters
        old_parameters = LearningParameters(
            learning_rate=self.parameters.learning_rate,
            utility_weights=self.parameters.utility_weights.copy(),
            quality_weights=self.parameters.quality_weights.copy(),
            discount_factor=self.parameters.discount_factor
        )
        
        self.parameters.update(gradients, adaptive_learning_rate)
        
        # Compute convergence metric
        convergence_metric = self._compute_convergence_metric(old_parameters)
        
        # Record learning step
        record = LearningRecord(
            timestamp=time.time(),
            parameters=LearningParameters(
                learning_rate=self.parameters.learning_rate,
                utility_weights=self.parameters.utility_weights.copy(),
                quality_weights=self.parameters.quality_weights.copy(),
                discount_factor=self.parameters.discount_factor
            ),
            performance_metrics=performance_metrics,
            gradients=gradients,
            learning_rate=adaptive_learning_rate,
            convergence_metric=convergence_metric
        )
        
        self.learning_history.append(record)
        self._save_data()
        
        return {
            "convergence_metric": convergence_metric,
            "learning_rate": adaptive_learning_rate,
            "gradients": gradients,
            "has_converged": convergence_metric < self.convergence_threshold
        }
    
    def _compute_convergence_metric(self, old_parameters: LearningParameters) -> float:
        """
        Compute convergence metric based on parameter changes.
        
        This implements the convergence analysis from the paper.
        """
        # Compute parameter change magnitude
        utility_change = sum(
            (self.parameters.utility_weights[key] - old_parameters.utility_weights[key])**2
            for key in self.parameters.utility_weights
        )
        
        quality_change = sum(
            (self.parameters.quality_weights[key] - old_parameters.quality_weights[key])**2
            for key in self.parameters.quality_weights
        )
        
        total_change = math.sqrt(utility_change + quality_change)
        return total_change
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence properties of the learning algorithm.
        
        Implements the convergence analysis from the paper.
        """
        if len(self.learning_history) < 2:
            return {
                "converged": False,
                "iterations": 0,
                "final_convergence_metric": 0.0,
                "convergence_rate": 0.0
            }
        
        # Check if algorithm has converged
        recent_metrics = [record.convergence_metric for record in self.learning_history[-10:]]
        has_converged = all(metric < self.convergence_threshold for metric in recent_metrics)
        
        # Compute convergence rate
        if len(self.learning_history) >= 2:
            initial_metric = self.learning_history[0].convergence_metric
            final_metric = self.learning_history[-1].convergence_metric
            convergence_rate = (initial_metric - final_metric) / len(self.learning_history)
        else:
            convergence_rate = 0.0
        
        return {
            "converged": has_converged,
            "iterations": len(self.learning_history),
            "final_convergence_metric": self.learning_history[-1].convergence_metric,
            "convergence_rate": convergence_rate,
            "convergence_threshold": self.convergence_threshold
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning process."""
        if not self.learning_history:
            return {
                "total_iterations": 0,
                "current_parameters": self.parameters.to_dict(),
                "convergence_status": "not_started"
            }
        
        convergence_analysis = self.analyze_convergence()
        
        # Compute learning statistics
        learning_rates = [record.learning_rate for record in self.learning_history]
        convergence_metrics = [record.convergence_metric for record in self.learning_history]
        
        return {
            "total_iterations": len(self.learning_history),
            "current_parameters": self.parameters.to_dict(),
            "convergence_status": convergence_analysis,
            "average_learning_rate": sum(learning_rates) / len(learning_rates),
            "average_convergence_metric": sum(convergence_metrics) / len(convergence_metrics),
            "learning_progress": {
                "initial_convergence": convergence_metrics[0],
                "current_convergence": convergence_metrics[-1],
                "improvement": convergence_metrics[0] - convergence_metrics[-1]
            }
        }
    
    def optimize_for_performance(self, 
                                performance_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Optimize parameters for better performance.
        
        This implements the optimization objective from the paper:
        J(π) = 𝔼_π[Σ_{t=0}^∞ γ^t R(s_t, a_t)]
        """
        if not performance_data:
            return {"optimized": False, "reason": "No performance data available"}
        
        # Compute average performance metrics
        avg_performance = {}
        for key in performance_data[0].keys():
            values = [data.get(key, 0.0) for data in performance_data]
            avg_performance[key] = sum(values) / len(values)
        
        # Update parameters based on performance
        update_result = self.update_parameters(avg_performance)
        
        return {
            "optimized": True,
            "update_result": update_result,
            "average_performance": avg_performance,
            "new_parameters": self.parameters.to_dict()
        }
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis."""
        return {
            "parameters": self.parameters.to_dict(),
            "learning_history": [record.to_dict() for record in self.learning_history],
            "convergence_analysis": self.analyze_convergence(),
            "learning_summary": self.get_learning_summary(),
            "export_timestamp": time.time()
        }
    
    def reset(self):
        """Reset the learner to initial state (useful for testing)."""
        self.learning_history = []
        self.parameters = LearningParameters()
    
    def _save_data(self):
        """Save learning data to file."""
        try:
            data = self.export_learning_data()
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save learning data: {e}")
    
    def _load_data(self):
        """Load learning data from file."""
        try:
            if Path(self.data_file).exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load parameters
                if "parameters" in data:
                    param_data = data["parameters"]
                    self.parameters = LearningParameters(
                        learning_rate=param_data.get("learning_rate", 0.01),
                        utility_weights=param_data.get("utility_weights", {}),
                        quality_weights=param_data.get("quality_weights", {}),
                        discount_factor=param_data.get("discount_factor", 0.9)
                    )
                
                # Load learning history
                self.learning_history = []
                for record_data in data.get("learning_history", []):
                    parameters = LearningParameters(
                        learning_rate=record_data["parameters"]["learning_rate"],
                        utility_weights=record_data["parameters"]["utility_weights"],
                        quality_weights=record_data["parameters"]["quality_weights"],
                        discount_factor=record_data["parameters"]["discount_factor"]
                    )
                    
                    record = LearningRecord(
                        timestamp=record_data["timestamp"],
                        parameters=parameters,
                        performance_metrics=record_data["performance_metrics"],
                        gradients=record_data["gradients"],
                        learning_rate=record_data["learning_rate"],
                        convergence_metric=record_data["convergence_metric"]
                    )
                    self.learning_history.append(record)
                
        except Exception as e:
            print(f"Warning: Could not load learning data: {e}")
            self.learning_history = [] 