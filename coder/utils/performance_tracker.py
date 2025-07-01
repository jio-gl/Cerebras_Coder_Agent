"""
Performance Tracking Framework implementing the metrics system from the paper.

This module implements the performance metrics framework described in the paper:
- Code Quality Score = α * Correctness + β * Efficiency + γ * Readability
- Performance tracking and analysis
- Historical data management
- Adaptive learning metrics
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class CodeQualityMetrics:
    """Represents code quality metrics as defined in the paper."""
    correctness: float = 0.0  # α weight
    efficiency: float = 0.0   # β weight
    readability: float = 0.0  # γ weight
    
    def compute_score(self, weights: Dict[str, float]) -> float:
        """
        Compute the overall code quality score.
        
        Implements the formula from the paper:
        Code Quality Score = α * Correctness + β * Efficiency + γ * Readability
        """
        alpha = weights.get("correctness", 0.4)
        beta = weights.get("efficiency", 0.3)
        gamma = weights.get("readability", 0.3)
        
        return (alpha * self.correctness + 
                beta * self.efficiency + 
                gamma * self.readability)


@dataclass
class PerformanceRecord:
    """Represents a single performance record."""
    timestamp: float
    operation: str
    execution_time: float
    success: bool
    quality_metrics: Optional[CodeQualityMetrics] = None
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "execution_time": self.execution_time,
            "success": self.success,
            "quality_metrics": {
                "correctness": self.quality_metrics.correctness if self.quality_metrics else 0.0,
                "efficiency": self.quality_metrics.efficiency if self.quality_metrics else 0.0,
                "readability": self.quality_metrics.readability if self.quality_metrics else 0.0
            } if self.quality_metrics else None,
            "error_message": self.error_message,
            "context": self.context
        }


class PerformanceTracker:
    """
    Implements the performance tracking framework from the paper.
    
    Tracks multiple performance metrics and provides analysis capabilities
    for the self-improvement cycle.
    """
    
    def __init__(self, data_file: Optional[str] = None, load_existing_data: bool = True):
        """
        Initialize the performance tracker.
        
        Args:
            data_file: Optional file path for persistent storage
            load_existing_data: Whether to load existing data from file (default: True)
        """
        self.data_file = data_file or "performance_data.json"
        self.records: List[PerformanceRecord] = []
        self.quality_weights = {
            "correctness": 0.4,
            "efficiency": 0.3,
            "readability": 0.3
        }
        
        # Load existing data if available and requested
        if load_existing_data:
            self._load_data()
    
    def record_operation(self, 
                        operation: str, 
                        execution_time: float, 
                        success: bool,
                        quality_metrics: Optional[CodeQualityMetrics] = None,
                        error_message: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> PerformanceRecord:
        """
        Record a performance measurement.
        
        Args:
            operation: Name of the operation performed
            execution_time: Time taken for the operation
            success: Whether the operation was successful
            quality_metrics: Optional quality metrics for the operation
            error_message: Optional error message if operation failed
            context: Optional context information
            
        Returns:
            The created performance record
        """
        record = PerformanceRecord(
            timestamp=time.time(),
            operation=operation,
            execution_time=execution_time,
            success=success,
            quality_metrics=quality_metrics,
            error_message=error_message,
            context=context or {}
        )
        
        self.records.append(record)
        self._save_data()
        
        return record
    
    def get_performance_summary(self, 
                               operation: Optional[str] = None,
                               time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Get a performance summary for analysis.
        
        Args:
            operation: Optional filter for specific operation
            time_window: Optional time window in seconds
            
        Returns:
            Dictionary containing performance statistics
        """
        # Filter records
        filtered_records = self.records
        
        if operation:
            filtered_records = [r for r in filtered_records if r.operation == operation]
        
        if time_window:
            cutoff_time = time.time() - time_window
            filtered_records = [r for r in filtered_records if r.timestamp >= cutoff_time]
        
        if not filtered_records:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "average_quality_score": 0.0,
                "error_rate": 0.0
            }
        
        # Calculate statistics
        total_operations = len(filtered_records)
        successful_operations = len([r for r in filtered_records if r.success])
        success_rate = successful_operations / total_operations
        
        execution_times = [r.execution_time for r in filtered_records]
        average_execution_time = sum(execution_times) / len(execution_times)
        
        # Calculate quality scores
        quality_scores = []
        for record in filtered_records:
            if record.quality_metrics:
                score = record.quality_metrics.compute_score(self.quality_weights)
                quality_scores.append(score)
        
        average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Error analysis
        error_operations = len([r for r in filtered_records if not r.success])
        error_rate = error_operations / total_operations
        
        return {
            "total_operations": total_operations,
            "success_rate": success_rate,
            "average_execution_time": average_execution_time,
            "average_quality_score": average_quality_score,
            "error_rate": error_rate,
            "quality_weights": self.quality_weights.copy()
        }
    
    def analyze_trends(self, operation: str, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific operation.
        
        Args:
            operation: Operation to analyze
            window_size: Number of recent records to consider
            
        Returns:
            Dictionary containing trend analysis
        """
        operation_records = [r for r in self.records if r.operation == operation]
        
        if len(operation_records) < window_size:
            return {"trend": "insufficient_data", "message": "Not enough data for trend analysis"}
        
        # Get recent records
        recent_records = operation_records[-window_size:]
        
        # Analyze execution time trend
        execution_times = [r.execution_time for r in recent_records]
        time_trend = self._calculate_trend(execution_times, lower_is_better=True)
        
        # Analyze success rate trend
        success_rates = []
        for i in range(0, len(recent_records), max(1, len(recent_records) // 5)):
            batch = recent_records[i:i+max(1, len(recent_records) // 5)]
            success_rate = len([r for r in batch if r.success]) / len(batch)
            success_rates.append(success_rate)
        
        success_trend = self._calculate_trend(success_rates)
        
        # Analyze quality score trend
        quality_scores = []
        for record in recent_records:
            if record.quality_metrics:
                score = record.quality_metrics.compute_score(self.quality_weights)
                quality_scores.append(score)
        
        quality_trend = self._calculate_trend(quality_scores) if quality_scores else "insufficient_data"
        
        return {
            "operation": operation,
            "window_size": window_size,
            "execution_time_trend": time_trend,
            "success_rate_trend": success_trend,
            "quality_score_trend": quality_trend,
            "recent_performance": self.get_performance_summary(operation)
        }
    
    def _calculate_trend(self, values: List[float], lower_is_better: bool = False) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        # For execution time, lower is better (improving)
        # For success rate and quality, higher is better (improving)
        if lower_is_better:
            # For execution time: lower values are better
            if second_avg < first_avg * 0.9:
                return "improving"
            elif second_avg > first_avg * 1.1:
                return "declining"
            else:
                return "stable"
        else:
            # For success rate and quality: higher values are better
            if second_avg > first_avg * 1.1:
                return "improving"
            elif second_avg < first_avg * 0.9:
                return "declining"
            else:
                return "stable"
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Generate insights for the adaptive learning mechanism.
        
        Returns insights that can be used to update model parameters
        as described in the paper: θ_{t+1} = θ_t + η_t ∇_θ J(θ_t)
        """
        if not self.records:
            return {"insights": [], "recommendations": []}
        
        insights = []
        recommendations = []
        
        # Analyze overall performance
        summary = self.get_performance_summary()
        
        if summary["success_rate"] < 0.8:
            insights.append("Low success rate detected")
            recommendations.append("Consider improving error handling and validation")
        
        if summary["average_execution_time"] > 5.0:
            insights.append("High execution time detected")
            recommendations.append("Consider optimizing tool selection and execution")
        
        if summary["average_quality_score"] < 0.7:
            insights.append("Low code quality scores detected")
            recommendations.append("Consider improving code generation quality")
        
        # Analyze operation-specific trends
        operations = set(r.operation for r in self.records)
        for operation in operations:
            trend = self.analyze_trends(operation)
            if trend.get("execution_time_trend") == "declining":
                insights.append(f"Performance declining for {operation}")
                recommendations.append(f"Review and optimize {operation} implementation")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "overall_performance": summary
        }
    
    def export_historical_data(self) -> Dict[str, Any]:
        """
        Export historical performance data for analysis.
        
        Returns data that can be used for the self-improvement cycle.
        """
        return {
            "records": [record.to_dict() for record in self.records],
            "quality_weights": self.quality_weights,
            "export_timestamp": time.time(),
            "total_records": len(self.records)
        }
    
    def reset(self):
        """Reset the tracker to initial state (useful for testing)."""
        self.records = []
    
    def _save_data(self):
        """Save performance data to file."""
        try:
            data = self.export_historical_data()
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Log error but don't fail
            print(f"Warning: Could not save performance data: {e}")
    
    def _load_data(self):
        """Load performance data from file."""
        try:
            if Path(self.data_file).exists():
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct records
                self.records = []
                for record_data in data.get("records", []):
                    quality_metrics = None
                    if record_data.get("quality_metrics"):
                        qm_data = record_data["quality_metrics"]
                        quality_metrics = CodeQualityMetrics(
                            correctness=qm_data.get("correctness", 0.0),
                            efficiency=qm_data.get("efficiency", 0.0),
                            readability=qm_data.get("readability", 0.0)
                        )
                    
                    record = PerformanceRecord(
                        timestamp=record_data["timestamp"],
                        operation=record_data["operation"],
                        execution_time=record_data["execution_time"],
                        success=record_data["success"],
                        quality_metrics=quality_metrics,
                        error_message=record_data.get("error_message"),
                        context=record_data.get("context", {})
                    )
                    self.records.append(record)
                
                # Load quality weights
                self.quality_weights = data.get("quality_weights", self.quality_weights)
                
        except Exception as e:
            # Log error but don't fail
            print(f"Warning: Could not load performance data: {e}")
            self.records = [] 