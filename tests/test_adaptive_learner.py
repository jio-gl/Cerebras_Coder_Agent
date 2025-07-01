"""
Tests for the Adaptive Learning Mechanism implementing the learning framework from the paper.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from coder.utils.adaptive_learner import (
    AdaptiveLearner, LearningParameters, LearningRecord
)


class TestLearningParameters:
    """Test the LearningParameters class."""
    
    def test_parameters_creation(self):
        """Test basic parameters creation."""
        params = LearningParameters(
            learning_rate=0.01,
            utility_weights={"relevance": 0.3, "success_rate": 0.2},
            quality_weights={"correctness": 0.4, "efficiency": 0.3},
            discount_factor=0.9
        )
        
        assert params.learning_rate == 0.01
        assert params.utility_weights["relevance"] == 0.3
        assert params.quality_weights["correctness"] == 0.4
        assert params.discount_factor == 0.9
    
    def test_parameters_defaults(self):
        """Test parameters with default values."""
        params = LearningParameters()
        
        assert params.learning_rate == 0.01
        assert "relevance" in params.utility_weights
        assert "correctness" in params.quality_weights
        assert params.discount_factor == 0.9
    
    def test_parameters_to_dict(self):
        """Test parameters serialization to dictionary."""
        params = LearningParameters(
            learning_rate=0.02,
            utility_weights={"relevance": 0.4, "success_rate": 0.3},
            quality_weights={"correctness": 0.5, "efficiency": 0.4},
            discount_factor=0.8
        )
        
        params_dict = params.to_dict()
        
        assert params_dict["learning_rate"] == 0.02
        assert params_dict["utility_weights"]["relevance"] == 0.4
        assert params_dict["quality_weights"]["correctness"] == 0.5
        assert params_dict["discount_factor"] == 0.8
    
    def test_parameters_update(self):
        """Test parameter update mechanism."""
        params = LearningParameters()
        initial_weights = params.utility_weights.copy()
        
        gradients = {"relevance": 0.1, "success_rate": -0.05}
        learning_rate = 0.01
        
        params.update(gradients, learning_rate)
        
        # Check that weights were updated
        assert params.utility_weights["relevance"] != initial_weights["relevance"]
        assert params.utility_weights["success_rate"] != initial_weights["success_rate"]
        
        # Check that weights stay within bounds
        for weight in params.utility_weights.values():
            assert 0.0 <= weight <= 1.0
    
    def test_parameters_normalization(self):
        """Test weight normalization."""
        params = LearningParameters()
        
        # Set weights to extreme values
        params.utility_weights = {"relevance": 2.0, "success_rate": 3.0}
        params.quality_weights = {"correctness": 1.5, "efficiency": 2.5}
        
        # Update with zero gradients to trigger normalization
        params.update({}, 0.0)
        
        # Check that weights are normalized (sum to 1)
        utility_sum = sum(params.utility_weights.values())
        quality_sum = sum(params.quality_weights.values())
        
        assert abs(utility_sum - 1.0) < 0.001
        assert abs(quality_sum - 1.0) < 0.001


class TestLearningRecord:
    """Test the LearningRecord class."""
    
    def test_record_creation(self):
        """Test basic record creation."""
        params = LearningParameters()
        performance_metrics = {"success_rate": 0.8, "efficiency": 0.7}
        gradients = {"relevance": 0.1, "success_rate": -0.05}
        
        record = LearningRecord(
            timestamp=1234567890.0,
            parameters=params,
            performance_metrics=performance_metrics,
            gradients=gradients,
            learning_rate=0.01,
            convergence_metric=0.001
        )
        
        assert record.timestamp == 1234567890.0
        assert record.parameters == params
        assert record.performance_metrics == performance_metrics
        assert record.gradients == gradients
        assert record.learning_rate == 0.01
        assert record.convergence_metric == 0.001
    
    def test_record_to_dict(self):
        """Test record serialization to dictionary."""
        params = LearningParameters()
        performance_metrics = {"success_rate": 0.8}
        gradients = {"relevance": 0.1}
        
        record = LearningRecord(
            timestamp=1234567890.0,
            parameters=params,
            performance_metrics=performance_metrics,
            gradients=gradients,
            learning_rate=0.01,
            convergence_metric=0.001
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["timestamp"] == 1234567890.0
        assert record_dict["learning_rate"] == 0.01
        assert record_dict["convergence_metric"] == 0.001
        assert record_dict["performance_metrics"] == performance_metrics
        assert record_dict["gradients"] == gradients
        assert "parameters" in record_dict


class TestAdaptiveLearner:
    """Test the AdaptiveLearner class."""
    
    def test_learner_initialization(self):
        """Test learner initialization."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        assert learner.parameters.learning_rate == 0.01
        assert len(learner.learning_history) == 0
        assert learner.convergence_threshold == 0.001
        assert learner.max_iterations == 1000
    
    def test_learner_initialization_custom_parameters(self):
        """Test learner initialization with custom parameters."""
        custom_params = LearningParameters(learning_rate=0.02, discount_factor=0.8)
        learner = AdaptiveLearner(initial_parameters=custom_params, load_existing_data=False)
        
        assert learner.parameters.learning_rate == 0.02
        assert learner.parameters.discount_factor == 0.8
    
    def test_learner_initialization_custom_file(self):
        """Test learner initialization with custom data file."""
        learner = AdaptiveLearner(data_file="custom_learning.json", load_existing_data=False)
        
        assert learner.data_file == "custom_learning.json"
    
    def test_compute_gradients(self):
        """Test gradient computation."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        current_performance = {
            "relevance": 0.7,
            "success_rate": 0.6,
            "quality_correctness": 0.8,
            "quality_efficiency": 0.7
        }
        
        target_performance = {
            "relevance": 0.9,
            "success_rate": 0.8,
            "quality_correctness": 0.9,
            "quality_efficiency": 0.8
        }
        
        gradients = learner.compute_gradients(current_performance, target_performance)
        
        # Check that gradients are computed correctly
        assert gradients["relevance"] == 0.2  # 0.9 - 0.7
        assert gradients["success_rate"] == 0.2  # 0.8 - 0.6
        assert gradients["correctness"] == 0.1  # 0.9 - 0.8
        assert gradients["efficiency"] == 0.1  # 0.8 - 0.7
    
    def test_compute_gradients_default_targets(self):
        """Test gradient computation with default targets."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        current_performance = {
            "relevance": 0.7,
            "success_rate": 0.6
        }
        
        gradients = learner.compute_gradients(current_performance, {})
        
        # Default target is 0.8 for all metrics
        assert gradients["relevance"] == 0.1  # 0.8 - 0.7
        assert gradients["success_rate"] == 0.2  # 0.8 - 0.6
    
    def test_update_parameters(self):
        """Test parameter update mechanism."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        initial_params = LearningParameters(
            learning_rate=learner.parameters.learning_rate,
            utility_weights=learner.parameters.utility_weights.copy(),
            quality_weights=learner.parameters.quality_weights.copy(),
            discount_factor=learner.parameters.discount_factor
        )
        
        performance_metrics = {
            "success_rate": 0.7,
            "average_quality_score": 0.6,
            "efficiency": 0.5
        }
        
        result = learner.update_parameters(performance_metrics)
        
        # Check that parameters were updated
        assert learner.parameters.utility_weights != initial_params.utility_weights
        assert learner.parameters.quality_weights != initial_params.quality_weights
        
        # Check result structure
        assert "convergence_metric" in result
        assert "learning_rate" in result
        assert "gradients" in result
        assert "has_converged" in result
        
        # Check that learning history was updated
        assert len(learner.learning_history) == 1
    
    def test_compute_convergence_metric(self):
        """Test convergence metric computation."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        old_params = LearningParameters(
            utility_weights={"relevance": 0.3, "success_rate": 0.2},
            quality_weights={"correctness": 0.4, "efficiency": 0.3}
        )
        
        # Update parameters to create a difference
        learner.parameters.utility_weights = {"relevance": 0.4, "success_rate": 0.3}
        learner.parameters.quality_weights = {"correctness": 0.5, "efficiency": 0.4}
        
        convergence_metric = learner._compute_convergence_metric(old_params)
        
        # Should be positive since parameters changed
        assert convergence_metric > 0
        assert isinstance(convergence_metric, float)
    
    def test_analyze_convergence_no_history(self):
        """Test convergence analysis with no history."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        analysis = learner.analyze_convergence()
        
        assert analysis["converged"] == False
        assert analysis["iterations"] == 0
        assert analysis["final_convergence_metric"] == 0.0
        assert analysis["convergence_rate"] == 0.0
    
    def test_analyze_convergence_with_history(self):
        """Test convergence analysis with learning history."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Add some learning records
        for i in range(15):
            performance_metrics = {"success_rate": 0.7 + (i * 0.01)}
            learner.update_parameters(performance_metrics)
        
        analysis = learner.analyze_convergence()
        
        assert analysis["iterations"] == 15
        assert "final_convergence_metric" in analysis
        assert "convergence_rate" in analysis
        assert "convergence_threshold" in analysis
    
    def test_get_learning_summary_no_history(self):
        """Test learning summary with no history."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        summary = learner.get_learning_summary()
        
        assert summary["total_iterations"] == 0
        assert summary["convergence_status"] == "not_started"
        assert "current_parameters" in summary
    
    def test_get_learning_summary_with_history(self):
        """Test learning summary with learning history."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Add some learning records
        for i in range(5):
            performance_metrics = {"success_rate": 0.7 + (i * 0.01)}
            learner.update_parameters(performance_metrics)
        
        summary = learner.get_learning_summary()
        
        assert summary["total_iterations"] == 5
        assert "convergence_status" in summary
        assert "average_learning_rate" in summary
        assert "average_convergence_metric" in summary
        assert "learning_progress" in summary
    
    def test_optimize_for_performance_no_data(self):
        """Test optimization with no performance data."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        result = learner.optimize_for_performance([])
        
        assert result["optimized"] == False
        assert "No performance data" in result["reason"]
    
    def test_optimize_for_performance_with_data(self):
        """Test optimization with performance data."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        performance_data = [
            {"success_rate": 0.7, "efficiency": 0.6},
            {"success_rate": 0.8, "efficiency": 0.7},
            {"success_rate": 0.9, "efficiency": 0.8}
        ]
        
        result = learner.optimize_for_performance(performance_data)
        
        assert result["optimized"] == True
        assert "update_result" in result
        assert "average_performance" in result
        assert "new_parameters" in result
        
        # Check average performance calculation
        avg_performance = result["average_performance"]
        assert abs(avg_performance["success_rate"] - 0.8) < 0.001  # (0.7 + 0.8 + 0.9) / 3
        assert abs(avg_performance["efficiency"] - 0.7) < 0.001  # (0.6 + 0.7 + 0.8) / 3
    
    def test_export_learning_data(self):
        """Test export of learning data."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Add some learning history
        performance_metrics = {"success_rate": 0.8}
        learner.update_parameters(performance_metrics)
        
        exported = learner.export_learning_data()
        
        assert "parameters" in exported
        assert "learning_history" in exported
        assert "convergence_analysis" in exported
        assert "learning_summary" in exported
        assert "export_timestamp" in exported
        
        assert len(exported["learning_history"]) == 1


class TestAdaptiveLearnerPersistence:
    """Test persistence functionality of the adaptive learner."""
    
    def test_save_and_load_data(self):
        """Test saving and loading data to/from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create learner with temp file
            learner = AdaptiveLearner(data_file=temp_file)
            
            # Add some learning history
            performance_metrics = {"success_rate": 0.8, "efficiency": 0.7}
            learner.update_parameters(performance_metrics)
            
            # Create new learner and load data
            new_learner = AdaptiveLearner(data_file=temp_file)
            
            # Verify data was loaded
            assert len(new_learner.learning_history) == 1
            assert new_learner.learning_history[0].performance_metrics == performance_metrics
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        learner = AdaptiveLearner(data_file="nonexistent_file.json")
        
        # Should not raise an error, just have empty history
        assert len(learner.learning_history) == 0
    
    def test_save_error_handling(self):
        """Test error handling during save."""
        learner = AdaptiveLearner(data_file="/invalid/path/learning.json")
        
        # Should not raise an error, just log warning
        performance_metrics = {"success_rate": 0.8}
        result = learner.update_parameters(performance_metrics)
        
        # Update should still work and record should be created in memory
        assert len(learner.learning_history) == 1
        assert "convergence_metric" in result


class TestAdaptiveLearnerIntegration:
    """Integration tests for the adaptive learner."""
    
    def test_complete_learning_workflow(self):
        """Test a complete learning workflow."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Simulate a learning process with improving performance
        performance_sequence = [
            {"success_rate": 0.6, "average_quality_score": 0.5, "efficiency": 0.4},
            {"success_rate": 0.7, "average_quality_score": 0.6, "efficiency": 0.5},
            {"success_rate": 0.8, "average_quality_score": 0.7, "efficiency": 0.6},
            {"success_rate": 0.85, "average_quality_score": 0.75, "efficiency": 0.65},
            {"success_rate": 0.9, "average_quality_score": 0.8, "efficiency": 0.7}
        ]
        
        # Update parameters based on performance
        for metrics in performance_sequence:
            result = learner.update_parameters(metrics)
            assert "convergence_metric" in result
            assert "learning_rate" in result
        
        # Check learning history
        assert len(learner.learning_history) == 5
        
        # Analyze convergence
        convergence = learner.analyze_convergence()
        assert convergence["iterations"] == 5
        
        # Get learning summary
        summary = learner.get_learning_summary()
        assert summary["total_iterations"] == 5
        assert "learning_progress" in summary
        
        # Export data
        exported = learner.export_learning_data()
        assert len(exported["learning_history"]) == 5
    
    def test_parameter_adaptation(self):
        """Test that parameters adapt based on performance."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        initial_weights = learner.parameters.utility_weights.copy()
        
        # Simulate poor performance in relevance
        poor_performance = {
            "relevance": 0.3,  # Low performance
            "success_rate": 0.8,  # Good performance
            "efficiency": 0.8  # Good performance
        }
        
        learner.update_parameters(poor_performance)
        
        # Check that relevance weight was adjusted
        assert learner.parameters.utility_weights["relevance"] != initial_weights["relevance"]
        
        # Check that other weights may have been adjusted too
        assert learner.parameters.utility_weights["success_rate"] != initial_weights["success_rate"]
    
    def test_convergence_detection(self):
        """Test convergence detection mechanism."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Add many updates with similar performance (should converge)
        for i in range(20):
            performance_metrics = {"success_rate": 0.8 + (i * 0.001)}  # Very small improvements
            result = learner.update_parameters(performance_metrics)
            
            # Check if convergence is detected
            if result["has_converged"]:
                break
        
        # Analyze convergence
        convergence = learner.analyze_convergence()
        
        # Should have some convergence analysis
        assert "converged" in convergence
        assert "iterations" in convergence
        assert "convergence_rate" in convergence
    
    def test_learning_rate_adaptation(self):
        """Test that learning rate adapts based on gradient magnitude."""
        learner = AdaptiveLearner(load_existing_data=False)
        
        # Large gradients should result in smaller learning rate
        large_gradient_performance = {
            "relevance": 0.1,  # Very poor performance (large gradient)
            "success_rate": 0.1,
            "efficiency": 0.1
        }
        
        result1 = learner.update_parameters(large_gradient_performance)
        
        # Small gradients should result in larger learning rate
        small_gradient_performance = {
            "relevance": 0.79,  # Close to target (small gradient)
            "success_rate": 0.79,
            "efficiency": 0.79
        }
        
        result2 = learner.update_parameters(small_gradient_performance)
        
        # Learning rates should be different
        assert result1["learning_rate"] != result2["learning_rate"] 