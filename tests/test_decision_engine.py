"""
Tests for the Decision Engine implementing MDP-based tool selection from the paper.
"""

import pytest
import time
from unittest.mock import Mock, patch

from coder.utils.decision_engine import (
    DecisionEngine, State, Tool, ToolType
)


class TestState:
    """Test the State class representing the coding environment state."""
    
    def test_state_creation(self):
        """Test basic state creation."""
        state = State(
            current_file="test.py",
            working_directory="/test",
            available_files=["test.py", "main.py"],
            task_description="analyze code"
        )
        
        assert state.current_file == "test.py"
        assert state.working_directory == "/test"
        assert state.available_files == ["test.py", "main.py"]
        assert state.task_description == "analyze code"
        assert state.timestamp > 0
    
    def test_state_to_dict(self):
        """Test state serialization to dictionary."""
        state = State(
            current_file="test.py",
            working_directory="/test",
            available_files=["test.py"],
            recent_actions=["read_file", "analyze_code"],
            task_description="test task"
        )
        
        state_dict = state.to_dict()
        
        assert state_dict["current_file"] == "test.py"
        assert state_dict["working_directory"] == "/test"
        assert state_dict["available_files"] == ["test.py"]
        assert state_dict["recent_actions"] == ["read_file", "analyze_code"]
        assert state_dict["task_description"] == "test task"
        assert "timestamp" in state_dict
    
    def test_state_recent_actions_limit(self):
        """Test that recent actions are limited to last 5."""
        state = State(
            recent_actions=["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        )
        
        state_dict = state.to_dict()
        assert len(state_dict["recent_actions"]) == 5
        assert state_dict["recent_actions"] == ["a3", "a4", "a5", "a6", "a7"]


class TestTool:
    """Test the Tool class representing available tools."""
    
    def test_tool_creation(self):
        """Test basic tool creation."""
        tool = Tool(
            name="test_tool",
            tool_type=ToolType.READ_FILE,
            description="Test tool description",
            parameters={"path": "string"},
            success_rate=0.9,
            execution_time=1.0,
            complexity=2.0
        )
        
        assert tool.name == "test_tool"
        assert tool.tool_type == ToolType.READ_FILE
        assert tool.description == "Test tool description"
        assert tool.parameters == {"path": "string"}
        assert tool.success_rate == 0.9
        assert tool.execution_time == 1.0
        assert tool.complexity == 2.0
    
    def test_tool_utility_computation(self):
        """Test utility computation for tools."""
        tool = Tool(
            name="read_file",
            tool_type=ToolType.READ_FILE,
            description="Read file",
            parameters={"path": "string"},
            success_rate=0.9,
            execution_time=0.1,
            complexity=1.0
        )
        
        state = State(
            task_description="read file",
            current_file="test.py",
            available_files=["test.py"]
        )
        
        weights = {
            "relevance": 0.3,
            "success_rate": 0.2,
            "efficiency": 0.2,
            "simplicity": 0.15,
            "context": 0.15
        }
        
        utility = tool.compute_utility(state, weights)
        assert utility > 0
        assert isinstance(utility, float)
    
    def test_tool_relevance_detection(self):
        """Test tool relevance detection for different tasks."""
        read_tool = Tool(
            name="read_file",
            tool_type=ToolType.READ_FILE,
            description="Read file",
            parameters={}
        )
        
        edit_tool = Tool(
            name="edit_file",
            tool_type=ToolType.EDIT_FILE,
            description="Edit file",
            parameters={}
        )
        
        analyze_tool = Tool(
            name="analyze_code",
            tool_type=ToolType.ANALYZE_CODE,
            description="Analyze code",
            parameters={}
        )
        
        # Test read task
        read_state = State(task_description="read the file")
        assert read_tool._is_relevant_to_task(read_state) == True
        assert edit_tool._is_relevant_to_task(read_state) == False
        
        # Test edit task
        edit_state = State(task_description="edit the code")
        assert edit_tool._is_relevant_to_task(edit_state) == True
        assert read_tool._is_relevant_to_task(edit_state) == False
        
        # Test analyze task
        analyze_state = State(task_description="analyze the code")
        assert analyze_tool._is_relevant_to_task(analyze_state) == True
    
    def test_tool_context_appropriateness(self):
        """Test tool context appropriateness."""
        read_tool = Tool(
            name="read_file",
            tool_type=ToolType.READ_FILE,
            description="Read file",
            parameters={}
        )
        
        edit_tool = Tool(
            name="edit_file",
            tool_type=ToolType.EDIT_FILE,
            description="Edit file",
            parameters={}
        )
        
        fix_tool = Tool(
            name="fix_syntax",
            tool_type=ToolType.FIX_SYNTAX,
            description="Fix syntax",
            parameters={}
        )
        
        # Test read tool with files available
        state_with_files = State(available_files=["test.py"])
        assert read_tool._is_context_appropriate(state_with_files) == True
        
        # Test edit tool with current file
        state_with_current = State(current_file="test.py")
        assert edit_tool._is_context_appropriate(state_with_current) == True
        
        # Test fix tool with error context
        state_with_error = State(error_context="SyntaxError")
        assert fix_tool._is_context_appropriate(state_with_error) == True


class TestDecisionEngine:
    """Test the DecisionEngine implementing MDP-based tool selection."""
    
    def test_decision_engine_initialization(self):
        """Test decision engine initialization."""
        engine = DecisionEngine(discount_factor=0.8)
        
        assert engine.discount_factor == 0.8
        assert len(engine.tools) > 0
        assert "relevance" in engine.utility_weights
        assert len(engine.state_history) == 0
        assert len(engine.action_history) == 0
    
    def test_tool_initialization(self):
        """Test that all required tools are initialized."""
        engine = DecisionEngine()
        
        tool_names = [tool.name for tool in engine.tools]
        expected_tools = [
            "read_file", "edit_file", "list_directory", "analyze_code",
            "optimize_code", "generate_tests", "enhance_error_handling",
            "explain_code", "refactor_code", "fix_syntax"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names
    
    def test_tool_selection(self):
        """Test tool selection algorithm."""
        engine = DecisionEngine()
        
        state = State(
            task_description="read file",
            current_file="test.py",
            available_files=["test.py"]
        )
        
        selected_tool = engine.select_tool(state)
        
        assert selected_tool is not None
        assert isinstance(selected_tool, Tool)
        assert selected_tool.name in [tool.name for tool in engine.tools]
    
    def test_tool_selection_for_different_tasks(self):
        """Test tool selection for different types of tasks."""
        engine = DecisionEngine()
        
        # Test read task
        read_state = State(task_description="read the file")
        read_tool = engine.select_tool(read_state)
        assert read_tool.tool_type == ToolType.READ_FILE
        
        # Test edit task
        edit_state = State(task_description="edit the code")
        edit_tool = engine.select_tool(edit_state)
        assert edit_tool.tool_type == ToolType.EDIT_FILE
        
        # Test analyze task
        analyze_state = State(task_description="analyze the code")
        analyze_tool = engine.select_tool(analyze_state)
        assert analyze_tool.tool_type == ToolType.ANALYZE_CODE
    
    def test_state_update(self):
        """Test state update mechanism."""
        engine = DecisionEngine()
        
        state1 = State(task_description="task1")
        state2 = State(task_description="task2")
        tool = engine.tools[0]
        reward = 1.0
        
        engine.update_state(state1, tool, reward, state2)
        
        assert len(engine.state_history) == 1
        assert len(engine.action_history) == 1
        assert engine.state_history[0] == state1
        assert engine.action_history[0] == (tool, state1, reward)
    
    def test_reward_computation(self):
        """Test reward function computation."""
        engine = DecisionEngine()
        
        state = State(task_description="test")
        tool = engine.tools[0]
        
        # Test successful outcome
        success_outcome = {
            "success": True,
            "execution_time": 1.0,
            "quality_score": 0.8
        }
        success_reward = engine.compute_reward(state, tool, success_outcome)
        assert success_reward > 0
        
        # Test failed outcome
        failure_outcome = {
            "success": False,
            "execution_time": 2.0,
            "error": True
        }
        failure_reward = engine.compute_reward(state, tool, failure_outcome)
        assert failure_reward < 0
    
    def test_decision_summary(self):
        """Test decision summary generation."""
        engine = DecisionEngine()
        
        # No decisions yet
        summary = engine.get_decision_summary()
        assert summary["total_decisions"] == 0
        assert summary["average_reward"] == 0.0
        
        # Add some decisions
        state = State(task_description="test")
        tool = engine.tools[0]
        engine.update_state(state, tool, 1.0, state)
        engine.update_state(state, tool, 2.0, state)
        
        summary = engine.get_decision_summary()
        assert summary["total_decisions"] == 2
        assert summary["average_reward"] == 1.5
        assert tool.name in summary["tool_usage"]
        assert summary["tool_usage"][tool.name] == 2
    
    def test_state_export_import(self):
        """Test state export and import functionality."""
        engine = DecisionEngine()
        
        # Add some data
        state = State(task_description="test")
        tool = engine.tools[0]
        engine.update_state(state, tool, 1.0, state)
        
        # Export state
        exported = engine.export_state()
        
        # Create new engine and import
        new_engine = DecisionEngine()
        new_engine.import_state(exported)
        
        # Verify data was imported correctly
        assert len(new_engine.state_history) == 1
        assert len(new_engine.action_history) == 1
        assert new_engine.discount_factor == engine.discount_factor
        assert new_engine.utility_weights == engine.utility_weights
    
    def test_history_management(self):
        """Test that history is kept manageable."""
        engine = DecisionEngine()
        
        # Add more than 100 records
        state = State(task_description="test")
        tool = engine.tools[0]
        
        for i in range(150):
            engine.update_state(state, tool, float(i), state)
        
        # Should be limited to 50 records
        assert len(engine.state_history) <= 50
        assert len(engine.action_history) <= 50


class TestDecisionEngineIntegration:
    """Integration tests for the decision engine."""
    
    def test_complete_workflow(self):
        """Test a complete decision-making workflow."""
        engine = DecisionEngine()
        
        # Simulate a coding task workflow
        states = [
            State(task_description="explore the codebase", working_directory="/project"),
            State(task_description="read the main file", current_file="main.py"),
            State(task_description="analyze the code", current_file="main.py"),
            State(task_description="fix syntax errors", error_context="SyntaxError"),
            State(task_description="generate tests", current_file="main.py")
        ]
        
        selected_tools = []
        for state in states:
            tool = engine.select_tool(state)
            selected_tools.append(tool)
            
            # Simulate execution outcome
            outcome = {
                "success": True,
                "execution_time": 1.0,
                "quality_score": 0.8
            }
            reward = engine.compute_reward(state, tool, outcome)
            
            # Update state
            next_state = State(task_description=f"completed {tool.name}")
            engine.update_state(state, tool, reward, next_state)
        
        # Verify workflow
        assert len(selected_tools) == 5
        assert len(engine.state_history) == 5
        assert len(engine.action_history) == 5
        
        # Verify tool selection was appropriate
        assert selected_tools[0].tool_type == ToolType.LIST_DIRECTORY  # explore
        assert selected_tools[1].tool_type == ToolType.READ_FILE       # read
        assert selected_tools[2].tool_type == ToolType.ANALYZE_CODE    # analyze
        assert selected_tools[3].tool_type == ToolType.FIX_SYNTAX      # fix
        assert selected_tools[4].tool_type == ToolType.GENERATE_TESTS  # tests
    
    def test_learning_from_experience(self):
        """Test that the engine learns from experience."""
        engine = DecisionEngine()
        
        # Initial decision
        state = State(task_description="read file")
        initial_tool = engine.select_tool(state)
        
        # Record successful execution
        outcome = {"success": True, "execution_time": 0.5, "quality_score": 0.9}
        reward = engine.compute_reward(state, initial_tool, outcome)
        engine.update_state(state, initial_tool, reward, state)
        
        # Make another decision (should be influenced by experience)
        next_state = State(task_description="read another file")
        next_tool = engine.select_tool(next_state)
        
        # Both should be read_file tools
        assert initial_tool.tool_type == ToolType.READ_FILE
        assert next_tool.tool_type == ToolType.READ_FILE
        
        # Verify experience was recorded
        summary = engine.get_decision_summary()
        assert summary["total_decisions"] == 1
        assert summary["average_reward"] > 0 