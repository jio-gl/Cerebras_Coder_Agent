"""
Decision Engine implementing MDP-based tool selection from the paper.

This module implements the mathematical framework for AI coding agent decision-making
processes as described in the paper, including:
- Markov Decision Process (MDP) for tool selection
- Utility-based decision making
- State space representation
- Reward function computation
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ToolType(Enum):
    """Enumeration of available tool types."""
    READ_FILE = "read_file"
    EDIT_FILE = "edit_file"
    LIST_DIRECTORY = "list_directory"
    ANALYZE_CODE = "analyze_code"
    OPTIMIZE_CODE = "optimize_code"
    GENERATE_TESTS = "generate_tests"
    ENHANCE_ERROR_HANDLING = "enhance_error_handling"
    EXPLAIN_CODE = "explain_code"
    REFACTOR_CODE = "refactor_code"
    FIX_SYNTAX = "fix_syntax"


@dataclass
class State:
    """Represents the current state of the coding environment."""
    current_file: Optional[str] = None
    working_directory: str = ""
    available_files: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    error_context: Optional[str] = None
    task_description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "current_file": self.current_file,
            "working_directory": self.working_directory,
            "available_files": self.available_files,
            "recent_actions": self.recent_actions[-5:],  # Keep last 5 actions
            "error_context": self.error_context,
            "task_description": self.task_description,
            "timestamp": self.timestamp
        }


@dataclass
class Tool:
    """Represents a tool with its capabilities and metadata."""
    name: str
    tool_type: ToolType
    description: str
    parameters: Dict[str, Any]
    success_rate: float = 0.8
    execution_time: float = 1.0
    complexity: float = 1.0
    
    def compute_utility(self, state: State, weights: Dict[str, float]) -> float:
        """
        Compute utility of this tool for the given state.
        
        Implements the utility function from the paper:
        U(t_i, s_t) = Σ w_j * f_j(t_i, s_t)
        """
        utility = 0.0
        
        # Relevance to current task
        if self._is_relevant_to_task(state):
            utility += weights.get("relevance", 0.3) * 1.0
        
        # Success probability
        utility += weights.get("success_rate", 0.2) * self.success_rate
        
        # Efficiency (inverse of execution time)
        efficiency = 1.0 / (1.0 + self.execution_time)
        utility += weights.get("efficiency", 0.2) * efficiency
        
        # Simplicity (inverse of complexity)
        simplicity = 1.0 / (1.0 + self.complexity)
        utility += weights.get("simplicity", 0.15) * simplicity
        
        # Context appropriateness
        if self._is_context_appropriate(state):
            utility += weights.get("context", 0.15) * 1.0
        
        return utility
    
    def _is_relevant_to_task(self, state: State) -> bool:
        """Check if tool is relevant to current task."""
        task_lower = state.task_description.lower()
        
        if "read" in task_lower or "view" in task_lower or "show" in task_lower:
            return self.tool_type == ToolType.READ_FILE
        elif "edit" in task_lower or "modify" in task_lower or "create" in task_lower or "write" in task_lower:
            return self.tool_type == ToolType.EDIT_FILE
        elif "list" in task_lower or "explore" in task_lower or "browse" in task_lower:
            return self.tool_type == ToolType.LIST_DIRECTORY
        elif "analyze" in task_lower or "examine" in task_lower or "inspect" in task_lower:
            return self.tool_type == ToolType.ANALYZE_CODE
        elif "optimize" in task_lower or "improve" in task_lower or "enhance" in task_lower:
            return self.tool_type == ToolType.OPTIMIZE_CODE
        elif "test" in task_lower or "generate test" in task_lower:
            return self.tool_type == ToolType.GENERATE_TESTS
        elif "error" in task_lower or "fix" in task_lower or "correct" in task_lower:
            return self.tool_type in [ToolType.ENHANCE_ERROR_HANDLING, ToolType.FIX_SYNTAX]
        elif "explain" in task_lower or "describe" in task_lower:
            return self.tool_type == ToolType.EXPLAIN_CODE
        elif "refactor" in task_lower or "restructure" in task_lower:
            return self.tool_type == ToolType.REFACTOR_CODE
        
        return True  # Default to relevant
    
    def _is_context_appropriate(self, state: State) -> bool:
        """Check if tool is appropriate for current context."""
        if self.tool_type == ToolType.READ_FILE:
            return state.current_file is not None or len(state.available_files) > 0
        elif self.tool_type == ToolType.EDIT_FILE:
            return state.current_file is not None
        elif self.tool_type == ToolType.ANALYZE_CODE:
            return state.current_file is not None
        elif self.tool_type == ToolType.FIX_SYNTAX:
            return state.error_context is not None
        
        return True


class DecisionEngine:
    """
    Implements the MDP-based decision making framework from the paper.
    
    The agent's decision-making process is formalized as a Markov Decision Process (MDP)
    with the following components:
    - S: State space representing code states and context
    - A: Action space of possible coding operations
    - T: Transition function
    - R: Reward function
    - γ: Discount factor
    """
    
    def __init__(self, discount_factor: float = 0.9):
        """
        Initialize the decision engine.
        
        Args:
            discount_factor: Discount factor γ for future rewards
        """
        self.discount_factor = discount_factor
        self.state_history: List[State] = []
        self.action_history: List[Tuple[Tool, State, float]] = []  # (tool, state, reward)
        
        # Default utility weights
        self.utility_weights = {
            "relevance": 0.5,  # Increased weight for relevance
            "success_rate": 0.15,
            "efficiency": 0.15,
            "simplicity": 0.1,
            "context": 0.1
        }
        
        # Initialize available tools
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize the set of available tools."""
        return [
            Tool(
                name="read_file",
                tool_type=ToolType.READ_FILE,
                description="Read file contents",
                parameters={"path": "string"},
                success_rate=0.95,
                execution_time=0.1,
                complexity=1.0
            ),
            Tool(
                name="edit_file",
                tool_type=ToolType.EDIT_FILE,
                description="Edit or create file",
                parameters={"path": "string", "content": "string"},
                success_rate=0.85,
                execution_time=0.5,
                complexity=2.0
            ),
            Tool(
                name="list_directory",
                tool_type=ToolType.LIST_DIRECTORY,
                description="List directory contents",
                parameters={"path": "string"},
                success_rate=0.9,
                execution_time=0.1,
                complexity=1.0
            ),
            Tool(
                name="analyze_code",
                tool_type=ToolType.ANALYZE_CODE,
                description="Analyze code structure and quality",
                parameters={"code": "string"},
                success_rate=0.8,
                execution_time=2.0,
                complexity=3.0
            ),
            Tool(
                name="optimize_code",
                tool_type=ToolType.OPTIMIZE_CODE,
                description="Optimize code for performance/readability",
                parameters={"code": "string", "goal": "string"},
                success_rate=0.75,
                execution_time=3.0,
                complexity=4.0
            ),
            Tool(
                name="generate_tests",
                tool_type=ToolType.GENERATE_TESTS,
                description="Generate unit tests for code",
                parameters={"code": "string"},
                success_rate=0.7,
                execution_time=2.5,
                complexity=3.5
            ),
            Tool(
                name="enhance_error_handling",
                tool_type=ToolType.ENHANCE_ERROR_HANDLING,
                description="Add error handling to code",
                parameters={"code": "string"},
                success_rate=0.8,
                execution_time=1.5,
                complexity=2.5
            ),
            Tool(
                name="explain_code",
                tool_type=ToolType.EXPLAIN_CODE,
                description="Explain code functionality",
                parameters={"code": "string", "level": "string"},
                success_rate=0.9,
                execution_time=1.0,
                complexity=1.5
            ),
            Tool(
                name="refactor_code",
                tool_type=ToolType.REFACTOR_CODE,
                description="Refactor code according to goal",
                parameters={"code": "string", "goal": "string"},
                success_rate=0.7,
                execution_time=2.0,
                complexity=3.0
            ),
            Tool(
                name="fix_syntax",
                tool_type=ToolType.FIX_SYNTAX,
                description="Fix Python syntax errors",
                parameters={"code": "string"},
                success_rate=0.85,
                execution_time=1.0,
                complexity=2.0
            )
        ]
    
    def select_tool(self, state: State) -> Tool:
        """
        Select the optimal tool for the current state.
        
        Implements the tool selection algorithm from the paper:
        1. Initialize tool set T = {t_1, t_2, ..., t_n}
        2. Observe current state s_t
        3. For each tool t_i ∈ T:
           - Compute utility U(t_i, s_t) = Σ w_j * f_j(t_i, s_t)
        4. Select tool t* = argmax_{t_i} U(t_i, s_t)
        5. Execute t* and observe new state s_{t+1}
        """
        best_tool = None
        best_utility = -float('inf')
        
        for tool in self.tools:
            utility = tool.compute_utility(state, self.utility_weights)
            if utility > best_utility:
                best_utility = utility
                best_tool = tool
        
        return best_tool
    
    def update_state(self, state: State, action: Tool, reward: float, new_state: State):
        """
        Update the decision engine with new experience.
        
        This implements the learning component of the MDP framework.
        """
        self.state_history.append(state)
        self.action_history.append((action, state, reward))
        
        # Keep history manageable
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
            self.action_history = self.action_history[-50:]
    
    def compute_reward(self, state: State, action: Tool, outcome: Dict[str, Any]) -> float:
        """
        Compute reward for the action taken.
        
        Implements the reward function R: S × A → ℝ from the paper.
        """
        reward = 0.0
        
        # Success reward
        if outcome.get("success", False):
            reward += 1.0
        
        # Efficiency reward (negative for time taken)
        execution_time = outcome.get("execution_time", 0.0)
        reward -= execution_time * 0.1
        
        # Quality reward
        if "quality_score" in outcome:
            reward += outcome["quality_score"] * 0.5
        
        # Error penalty
        if outcome.get("error", False):
            reward -= 2.0
        
        return reward
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get a summary of decision-making performance."""
        if not self.action_history:
            return {"total_decisions": 0, "average_reward": 0.0}
        
        total_decisions = len(self.action_history)
        total_reward = sum(reward for _, _, reward in self.action_history)
        average_reward = total_reward / total_decisions
        
        # Tool usage statistics
        tool_usage = {}
        for tool, _, _ in self.action_history:
            tool_usage[tool.name] = tool_usage.get(tool.name, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "average_reward": average_reward,
            "tool_usage": tool_usage,
            "discount_factor": self.discount_factor
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "discount_factor": self.discount_factor,
            "utility_weights": self.utility_weights,
            "state_history": [state.to_dict() for state in self.state_history],
            "action_history": [
                {
                    "tool_name": tool.name,
                    "state": state.to_dict(),
                    "reward": reward
                }
                for tool, state, reward in self.action_history
            ]
        }
    
    def import_state(self, state_data: Dict[str, Any]):
        """Import state from persistence."""
        self.discount_factor = state_data.get("discount_factor", 0.9)
        self.utility_weights = state_data.get("utility_weights", self.utility_weights)
        
        # Reconstruct state history
        self.state_history = []
        for state_dict in state_data.get("state_history", []):
            state = State(**state_dict)
            self.state_history.append(state)
        
        # Reconstruct action history
        self.action_history = []
        for action_dict in state_data.get("action_history", []):
            tool_name = action_dict["tool_name"]
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                state = State(**action_dict["state"])
                reward = action_dict["reward"]
                self.action_history.append((tool, state, reward)) 