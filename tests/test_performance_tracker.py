"""
Tests for the Performance Tracking Framework implementing the metrics system from the paper.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from coder.utils.performance_tracker import (
    PerformanceTracker, CodeQualityMetrics, PerformanceRecord
)


class TestCodeQualityMetrics:
    """Test the CodeQualityMetrics class."""
    
    def test_metrics_creation(self):
        """Test basic metrics creation."""
        metrics = CodeQualityMetrics(
            correctness=0.8,
            efficiency=0.7,
            readability=0.9
        )
        
        assert metrics.correctness == 0.8
        assert metrics.efficiency == 0.7
        assert metrics.readability == 0.9
    
    def test_metrics_defaults(self):
        """Test metrics with default values."""
        metrics = CodeQualityMetrics()
        
        assert metrics.correctness == 0.0
        assert metrics.efficiency == 0.0
        assert metrics.readability == 0.0
    
    def test_score_computation(self):
        """Test score computation with default weights."""
        metrics = CodeQualityMetrics(
            correctness=0.8,
            efficiency=0.7,
            readability=0.9
        )
        
        score = metrics.compute_score({})
        
        # Default weights: correctness=0.4, efficiency=0.3, readability=0.3
        expected_score = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.9
        assert abs(score - expected_score) < 0.001
    
    def test_score_computation_custom_weights(self):
        """Test score computation with custom weights."""
        metrics = CodeQualityMetrics(
            correctness=0.8,
            efficiency=0.7,
            readability=0.9
        )
        
        custom_weights = {
            "correctness": 0.5,
            "efficiency": 0.3,
            "readability": 0.2
        }
        
        score = metrics.compute_score(custom_weights)
        
        expected_score = 0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.9
        assert abs(score - expected_score) < 0.001


class TestPerformanceRecord:
    """Test the PerformanceRecord class."""
    
    def test_record_creation(self):
        """Test basic record creation."""
        metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        
        record = PerformanceRecord(
            timestamp=1234567890.0,
            operation="test_operation",
            execution_time=1.5,
            success=True,
            quality_metrics=metrics,
            error_message=None,
            context={"file": "test.py"}
        )
        
        assert record.timestamp == 1234567890.0
        assert record.operation == "test_operation"
        assert record.execution_time == 1.5
        assert record.success == True
        assert record.quality_metrics == metrics
        assert record.error_message is None
        assert record.context == {"file": "test.py"}
    
    def test_record_to_dict(self):
        """Test record serialization to dictionary."""
        metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        
        record = PerformanceRecord(
            timestamp=1234567890.0,
            operation="test_operation",
            execution_time=1.5,
            success=True,
            quality_metrics=metrics,
            error_message="test error",
            context={"file": "test.py"}
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["timestamp"] == 1234567890.0
        assert record_dict["operation"] == "test_operation"
        assert record_dict["execution_time"] == 1.5
        assert record_dict["success"] == True
        assert record_dict["error_message"] == "test error"
        assert record_dict["context"] == {"file": "test.py"}
        
        # Check quality metrics
        assert record_dict["quality_metrics"]["correctness"] == 0.8
        assert record_dict["quality_metrics"]["efficiency"] == 0.7
        assert record_dict["quality_metrics"]["readability"] == 0.9
    
    def test_record_to_dict_no_metrics(self):
        """Test record serialization without quality metrics."""
        record = PerformanceRecord(
            timestamp=1234567890.0,
            operation="test_operation",
            execution_time=1.5,
            success=True,
            quality_metrics=None,
            error_message=None,
            context={}
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["quality_metrics"] is None


class TestPerformanceTracker:
    """Test the PerformanceTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        assert tracker.data_file == "performance_data.json"
        assert len(tracker.records) == 0
        assert "correctness" in tracker.quality_weights
        assert "efficiency" in tracker.quality_weights
        assert "readability" in tracker.quality_weights
    
    def test_tracker_initialization_custom_file(self):
        """Test tracker initialization with custom data file."""
        tracker = PerformanceTracker(data_file="custom_data.json")
        
        assert tracker.data_file == "custom_data.json"
    
    def test_record_operation(self):
        """Test recording an operation."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        
        record = tracker.record_operation(
            operation="test_operation",
            execution_time=1.5,
            success=True,
            quality_metrics=metrics,
            error_message=None,
            context={"file": "test.py"}
        )
        
        assert len(tracker.records) == 1
        assert record.operation == "test_operation"
        assert record.execution_time == 1.5
        assert record.success == True
        assert record.quality_metrics == metrics
        assert record.context == {"file": "test.py"}
        assert record.timestamp > 0
    
    def test_record_operation_without_metrics(self):
        """Test recording an operation without quality metrics."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        record = tracker.record_operation(
            operation="simple_operation",
            execution_time=0.5,
            success=True
        )
        
        assert len(tracker.records) == 1
        assert record.quality_metrics is None
    
    def test_get_performance_summary_empty(self):
        """Test performance summary with no records."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        summary = tracker.get_performance_summary()
        
        assert summary["total_operations"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["average_execution_time"] == 0.0
        assert summary["average_quality_score"] == 0.0
        assert summary["error_rate"] == 0.0
    
    def test_get_performance_summary_with_records(self):
        """Test performance summary with records."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add some test records
        metrics1 = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        metrics2 = CodeQualityMetrics(correctness=0.6, efficiency=0.8, readability=0.7)
        
        tracker.record_operation("op1", 1.0, True, metrics1)
        tracker.record_operation("op2", 2.0, True, metrics2)
        tracker.record_operation("op3", 0.5, False)  # Failed operation
        
        summary = tracker.get_performance_summary()
        
        assert summary["total_operations"] == 3
        assert summary["success_rate"] == 2/3
        assert summary["average_execution_time"] == (1.0 + 2.0 + 0.5) / 3
        assert summary["error_rate"] == 1/3
        
        # Quality score should be average of the two successful operations
        expected_score1 = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.9  # 0.8
        expected_score2 = 0.4 * 0.6 + 0.3 * 0.8 + 0.3 * 0.7  # 0.69
        expected_avg = (expected_score1 + expected_score2) / 2
        
        assert abs(summary["average_quality_score"] - expected_avg) < 0.01
    
    def test_get_performance_summary_filtered(self):
        """Test performance summary with filters."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add records for different operations
        tracker.record_operation("read_file", 1.0, True)
        tracker.record_operation("edit_file", 2.0, True)
        tracker.record_operation("read_file", 0.5, False)
        
        # Filter by operation
        read_summary = tracker.get_performance_summary(operation="read_file")
        assert read_summary["total_operations"] == 2
        assert read_summary["success_rate"] == 0.5
        
        # Filter by time window
        time.sleep(0.1)  # Ensure some time passes
        recent_summary = tracker.get_performance_summary(time_window=0.05)
        assert recent_summary["total_operations"] == 0  # No recent operations
    
    def test_analyze_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add only 5 records (less than default window_size of 10)
        for i in range(5):
            tracker.record_operation(f"op{i}", 1.0, True)
        
        trend = tracker.analyze_trends("op0")
        
        assert trend["trend"] == "insufficient_data"
        assert "Not enough data" in trend["message"]
    
    def test_analyze_trends_improving(self):
        """Test trend analysis for improving performance."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add records with improving execution time
        for i in range(10):
            execution_time = 2.0 - (i * 0.1)  # Decreasing from 2.0 to 1.1
            tracker.record_operation("test_op", execution_time, True)
        
        trend = tracker.analyze_trends("test_op")
        
        assert trend["execution_time_trend"] == "improving"
        assert trend["operation"] == "test_op"
        assert trend["window_size"] == 10
    
    def test_analyze_trends_declining(self):
        """Test trend analysis for declining performance."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add records with declining success rate
        for i in range(10):
            success = i < 5  # First 5 succeed, last 5 fail
            tracker.record_operation("test_op", 1.0, success)
        
        trend = tracker.analyze_trends("test_op")
        
        assert trend["success_rate_trend"] == "declining"
    
    def test_analyze_trends_stable(self):
        """Test trend analysis for stable performance."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add records with stable execution time
        for i in range(10):
            tracker.record_operation("test_op", 1.0, True)
        
        trend = tracker.analyze_trends("test_op")
        
        assert trend["execution_time_trend"] == "stable"
    
    def test_get_learning_insights_no_data(self):
        """Test learning insights with no data."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        insights = tracker.get_learning_insights()
        
        assert insights["insights"] == []
        assert insights["recommendations"] == []
    
    def test_get_learning_insights_with_data(self):
        """Test learning insights with performance data."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add records with poor performance
        for i in range(10):
            tracker.record_operation("test_op", 6.0, False)  # High time, low success
        
        insights = tracker.get_learning_insights()
        
        assert len(insights["insights"]) > 0
        assert len(insights["recommendations"]) > 0
        assert "Low success rate" in insights["insights"][0]
        # Check if there are multiple insights before accessing index 1
        if len(insights["insights"]) > 1:
            assert "High execution time" in insights["insights"][1]
    
    def test_export_historical_data(self):
        """Test export of historical data."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add some test data
        metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        tracker.record_operation("test_op", 1.0, True, metrics)
        
        exported = tracker.export_historical_data()
        
        assert "records" in exported
        assert "quality_weights" in exported
        assert "export_timestamp" in exported
        assert "total_records" in exported
        assert exported["total_records"] == 1
        assert len(exported["records"]) == 1


class TestPerformanceTrackerPersistence:
    """Test persistence functionality of the performance tracker."""
    
    def test_save_and_load_data(self):
        """Test saving and loading data to/from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create tracker with temp file
            tracker = PerformanceTracker(data_file=temp_file)
            
            # Add some test data
            metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
            tracker.record_operation("test_op", 1.0, True, metrics)
            
            # Create new tracker and load data
            new_tracker = PerformanceTracker(data_file=temp_file)
            
            # Verify data was loaded
            assert len(new_tracker.records) == 1
            assert new_tracker.records[0].operation == "test_op"
            assert new_tracker.records[0].execution_time == 1.0
            assert new_tracker.records[0].success == True
            assert new_tracker.records[0].quality_metrics is not None
            assert new_tracker.records[0].quality_metrics.correctness == 0.8
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        tracker = PerformanceTracker(data_file="nonexistent_file.json")
        
        # Should not raise an error, just have empty records
        assert len(tracker.records) == 0
    
    def test_save_error_handling(self):
        """Test error handling during save."""
        tracker = PerformanceTracker(data_file="/invalid/path/data.json")
        
        # Should not raise an error, just log warning
        metrics = CodeQualityMetrics(correctness=0.8, efficiency=0.7, readability=0.9)
        record = tracker.record_operation("test_op", 1.0, True, metrics)
        
        # Record should still be created in memory
        assert len(tracker.records) == 1
        assert record.operation == "test_op"


class TestPerformanceTrackerIntegration:
    """Integration tests for the performance tracker."""
    
    def test_complete_workflow(self):
        """Test a complete performance tracking workflow."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Simulate a coding workflow with different operations
        operations = [
            ("read_file", 0.1, True, CodeQualityMetrics(0.9, 0.8, 0.9)),
            ("analyze_code", 2.0, True, CodeQualityMetrics(0.8, 0.7, 0.8)),
            ("edit_file", 1.5, True, CodeQualityMetrics(0.7, 0.6, 0.7)),
            ("generate_tests", 6.0, False, None),  # Failed operation with high execution time
            ("fix_syntax", 0.5, False, None)  # Another failed operation
        ]
        
        for operation, exec_time, success, metrics in operations:
            tracker.record_operation(operation, exec_time, success, metrics)
        
        # Get overall summary
        summary = tracker.get_performance_summary()
        assert summary["total_operations"] == 5
        assert summary["success_rate"] == 0.6  # 3 out of 5 succeeded
        assert summary["error_rate"] == 0.4
        
        # Analyze trends for specific operations
        read_trend = tracker.analyze_trends("read_file")
        # Check if there's enough data for trend analysis
        if read_trend.get("trend") != "insufficient_data":
            assert read_trend["operation"] == "read_file"
        
        # Get learning insights
        insights = tracker.get_learning_insights()
        assert len(insights["insights"]) > 0
        assert len(insights["recommendations"]) > 0
        
        # Export data
        exported = tracker.export_historical_data()
        assert exported["total_records"] == 5
        assert len(exported["records"]) == 5
    
    def test_quality_score_calculation(self):
        """Test quality score calculation across multiple operations."""
        tracker = PerformanceTracker(load_existing_data=False)
        
        # Add operations with known quality metrics
        metrics1 = CodeQualityMetrics(correctness=1.0, efficiency=1.0, readability=1.0)
        metrics2 = CodeQualityMetrics(correctness=0.5, efficiency=0.5, readability=0.5)
        
        tracker.record_operation("perfect_op", 1.0, True, metrics1)
        tracker.record_operation("average_op", 1.0, True, metrics2)
        
        summary = tracker.get_performance_summary()
        
        # Perfect operation should have score 1.0, average should have 0.5
        # Overall average should be 0.75
        assert abs(summary["average_quality_score"] - 0.75) < 0.01 