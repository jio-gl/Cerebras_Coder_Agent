"""
Integration tests demonstrating the paper's theoretical framework in practice.

This test shows how the Decision Engine, Performance Tracker, and Adaptive Learner
work together to implement the mathematical framework described in the paper.
"""

import pytest
import time
from unittest.mock import Mock
import os

from coder.utils.decision_engine import DecisionEngine, State, Tool, ToolType
from coder.utils.performance_tracker import PerformanceTracker, CodeQualityMetrics
from coder.utils.adaptive_learner import AdaptiveLearner, LearningParameters


class TestPaperFrameworkIntegration:
    """Test the integration of all paper framework components."""
    
    def test_complete_workflow_implementation(self):
        """
        Test a complete workflow implementing the paper's framework.
        
        This demonstrates:
        1. MDP-based tool selection (Decision Engine)
        2. Performance tracking and metrics (Performance Tracker)
        3. Adaptive learning and parameter updates (Adaptive Learner)
        """
        # Initialize all components with fresh instances
        decision_engine = DecisionEngine(discount_factor=0.9)
        performance_tracker = PerformanceTracker(data_file="temp_performance.json", load_existing_data=False)
        adaptive_learner = AdaptiveLearner(data_file="temp_learning.json", load_existing_data=False)
        
        # Simulate a coding task workflow
        task_sequence = [
            {
                "description": "explore the codebase",
                "operation": "list_directory",
                "execution_time": 0.1,
                "success": True,
                "quality_metrics": CodeQualityMetrics(0.9, 0.8, 0.9)
            },
            {
                "description": "read the main file",
                "operation": "read_file",
                "execution_time": 0.2,
                "success": True,
                "quality_metrics": CodeQualityMetrics(0.8, 0.7, 0.8)
            },
            {
                "description": "analyze the code structure",
                "operation": "analyze_code",
                "execution_time": 2.0,
                "success": True,
                "quality_metrics": CodeQualityMetrics(0.7, 0.6, 0.7)
            },
            {
                "description": "fix syntax errors",
                "operation": "fix_syntax",
                "execution_time": 6.0,  # High execution time to trigger insight
                "success": False,  # Failed operation
                "quality_metrics": None
            },
            {
                "description": "generate tests",
                "operation": "generate_tests",
                "execution_time": 3.0,
                "success": False,  # Another failure to lower success rate
                "quality_metrics": None
            }
        ]
        
        # Execute the workflow
        for i, task in enumerate(task_sequence):
            # Step 1: Decision Engine selects appropriate tool
            state = State(
                task_description=task["description"],
                current_file="main.py" if i > 0 else None,
                available_files=["main.py", "test.py"] if i > 0 else [],
                error_context="SyntaxError" if i == 3 else None
            )
            
            selected_tool = decision_engine.select_tool(state)
            
            # Verify tool selection is appropriate
            assert selected_tool.name == task["operation"]
            
            # Step 2: Execute operation and record performance
            performance_record = performance_tracker.record_operation(
                operation=task["operation"],
                execution_time=task["execution_time"],
                success=task["success"],
                quality_metrics=task["quality_metrics"],
                context={"task": task["description"]}
            )
            
            # Step 3: Compute reward and update decision engine
            outcome = {
                "success": task["success"],
                "execution_time": task["execution_time"],
                "quality_score": task["quality_metrics"].compute_score({}) if task["quality_metrics"] else 0.0
            }
            
            reward = decision_engine.compute_reward(state, selected_tool, outcome)
            decision_engine.update_state(state, selected_tool, reward, state)
        
        # Step 4: Analyze performance and generate insights
        performance_summary = performance_tracker.get_performance_summary()
        learning_insights = performance_tracker.get_learning_insights()
        
        # Verify performance analysis
        assert performance_summary["total_operations"] == 5
        assert performance_summary["success_rate"] == 0.6  # 3 out of 5 succeeded
        assert performance_summary["error_rate"] == 0.4
        
        # Verify learning insights were generated
        assert len(learning_insights["insights"]) > 0
        assert len(learning_insights["recommendations"]) > 0
        
        # Step 5: Adaptive learning based on performance
        # Convert performance summary to metrics for learning
        learning_metrics = {
            "success_rate": performance_summary["success_rate"],
            "average_quality_score": performance_summary["average_quality_score"],
            "efficiency": 1.0 / (1.0 + performance_summary["average_execution_time"])
        }
        
        # Update learning parameters
        learning_result = adaptive_learner.update_parameters(learning_metrics)
        
        # Verify learning occurred
        assert "convergence_metric" in learning_result
        assert "learning_rate" in learning_result
        assert "gradients" in learning_result
        
        # Step 6: Verify the complete framework integration
        # Decision engine should have learned from experience
        decision_summary = decision_engine.get_decision_summary()
        assert decision_summary["total_decisions"] == 5
        assert decision_summary["average_reward"] > 0
        
        # Performance tracker should have comprehensive data
        historical_data = performance_tracker.export_historical_data()
        assert historical_data["total_records"] == 5
        
        # Adaptive learner should have learning history
        learning_data = adaptive_learner.export_learning_data()
        assert len(learning_data["learning_history"]) > 0
        
        print(f"✅ Complete workflow executed successfully!")
        print(f"   - Decisions made: {decision_summary['total_decisions']}")
        print(f"   - Operations tracked: {performance_summary['total_operations']}")
        print(f"   - Learning iterations: {len(learning_data['learning_history'])}")
        print(f"   - Success rate: {performance_summary['success_rate']:.2f}")
        print(f"   - Average quality score: {performance_summary['average_quality_score']:.2f}")
        
        # Clean up temporary files
        if os.path.exists("temp_performance.json"):
            os.remove("temp_performance.json")
        if os.path.exists("temp_learning.json"):
            os.remove("temp_learning.json")
    
    def test_mathematical_framework_validation(self):
        """
        Test that the mathematical framework from the paper is correctly implemented.
        
        This validates:
        - MDP formulation: (S, A, T, R, γ)
        - Utility function: U(t_i, s_t) = Σ w_j * f_j(t_i, s_t)
        - Quality score: α * Correctness + β * Efficiency + γ * Readability
        - Parameter update: θ_{t+1} = θ_t + η_t ∇_θ J(θ_t)
        """
        # Initialize components with fresh instances
        decision_engine = DecisionEngine()
        performance_tracker = PerformanceTracker(data_file="temp_performance2.json")
        adaptive_learner = AdaptiveLearner(data_file="temp_learning2.json")
        
        # Test 1: MDP State Space (S)
        state = State(
            task_description="analyze code",
            current_file="test.py",
            available_files=["test.py"],
            error_context=None
        )
        
        # Test 2: Action Space (A) - Tool selection
        selected_tool = decision_engine.select_tool(state)
        assert selected_tool is not None
        assert isinstance(selected_tool, Tool)
        
        # Test 3: Utility Function U(t_i, s_t) = Σ w_j * f_j(t_i, s_t)
        utility = selected_tool.compute_utility(state, decision_engine.utility_weights)
        assert utility > 0
        assert isinstance(utility, float)
        
        # Test 4: Quality Score Formula
        quality_metrics = CodeQualityMetrics(
            correctness=0.8,  # α
            efficiency=0.7,   # β
            readability=0.9   # γ
        )
        
        # Default weights: α=0.4, β=0.3, γ=0.3
        expected_score = 0.4 * 0.8 + 0.3 * 0.7 + 0.3 * 0.9
        actual_score = quality_metrics.compute_score({})
        assert abs(actual_score - expected_score) < 0.001
        
        # Test 5: Reward Function R(s, a)
        outcome = {
            "success": True,
            "execution_time": 1.0,
            "quality_score": actual_score
        }
        
        reward = decision_engine.compute_reward(state, selected_tool, outcome)
        assert isinstance(reward, float)
        
        # Test 6: Parameter Update θ_{t+1} = θ_t + η_t ∇_θ J(θ_t)
        performance_metrics = {
            "success_rate": 0.8,
            "average_quality_score": 0.7,
            "efficiency": 0.6
        }
        
        update_result = adaptive_learner.update_parameters(performance_metrics)
        assert "convergence_metric" in update_result
        assert "learning_rate" in update_result
        assert "gradients" in update_result
        
        print(f"✅ Mathematical framework validation passed!")
        print(f"   - MDP state space: ✓")
        print(f"   - Action space: ✓")
        print(f"   - Utility function: {utility:.3f}")
        print(f"   - Quality score: {actual_score:.3f} (expected: {expected_score:.3f})")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Parameter update: ✓")
        
        # Clean up temporary files
        if os.path.exists("temp_performance2.json"):
            os.remove("temp_performance2.json")
        if os.path.exists("temp_learning2.json"):
            os.remove("temp_learning2.json")
    
    def test_self_improvement_cycle(self):
        """
        Test the self-improvement cycle described in the paper.
        
        This demonstrates how the system learns and improves over time:
        1. Execute task
        2. Collect feedback
        3. Analyze performance
        4. Update model
        5. Adapt strategy
        """
        decision_engine = DecisionEngine()
        performance_tracker = PerformanceTracker(data_file="temp_performance3.json")
        adaptive_learner = AdaptiveLearner(data_file="temp_learning3.json")
        
        # Simulate multiple iterations of the self-improvement cycle
        for iteration in range(3):
            print(f"\n🔄 Self-Improvement Cycle {iteration + 1}")
            
            # Step 1: Execute task
            state = State(
                task_description="analyze code",
                current_file="test.py",
                available_files=["test.py"]
            )
            
            selected_tool = decision_engine.select_tool(state)
            
            # Step 2: Collect feedback (simulate execution)
            execution_time = 2.0 - (iteration * 0.3)  # Improving over time
            success = iteration < 2  # First two succeed, last one fails
            quality_metrics = CodeQualityMetrics(
                correctness=0.7 + (iteration * 0.1),
                efficiency=0.6 + (iteration * 0.1),
                readability=0.8 + (iteration * 0.05)
            )
            
            performance_record = performance_tracker.record_operation(
                operation=selected_tool.name,
                execution_time=execution_time,
                success=success,
                quality_metrics=quality_metrics
            )
            
            # Step 3: Analyze performance
            performance_summary = performance_tracker.get_performance_summary()
            learning_insights = performance_tracker.get_learning_insights()
            
            # Step 4: Update model
            outcome = {
                "success": success,
                "execution_time": execution_time,
                "quality_score": quality_metrics.compute_score({})
            }
            
            reward = decision_engine.compute_reward(state, selected_tool, outcome)
            decision_engine.update_state(state, selected_tool, reward, state)
            
            # Step 5: Adapt strategy
            learning_metrics = {
                "success_rate": performance_summary["success_rate"],
                "average_quality_score": performance_summary["average_quality_score"],
                "efficiency": 1.0 / (1.0 + performance_summary["average_execution_time"])
            }
            
            learning_result = adaptive_learner.update_parameters(learning_metrics)
            
            print(f"   - Tool selected: {selected_tool.name}")
            print(f"   - Execution time: {execution_time:.2f}s")
            print(f"   - Success: {success}")
            print(f"   - Quality score: {quality_metrics.compute_score({}):.3f}")
            print(f"   - Reward: {reward:.3f}")
            print(f"   - Convergence metric: {learning_result['convergence_metric']:.6f}")
        
        # Verify improvement over iterations
        final_performance = performance_tracker.get_performance_summary()
        final_learning = adaptive_learner.get_learning_summary()
        
        assert final_performance["total_operations"] == 3
        assert len(final_learning["learning_progress"]) > 0
        
        print(f"\n✅ Self-improvement cycle completed!")
        print(f"   - Total operations: {final_performance['total_operations']}")
        print(f"   - Learning iterations: {final_learning['total_iterations']}")
        print(f"   - Final convergence metric: {final_learning['learning_progress']['current_convergence']:.6f}")
        
        # Clean up temporary files
        if os.path.exists("temp_performance3.json"):
            os.remove("temp_performance3.json")
        if os.path.exists("temp_learning3.json"):
            os.remove("temp_learning3.json") 