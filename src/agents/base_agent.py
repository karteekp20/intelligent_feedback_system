"""
Base agent class for the Intelligent Feedback Analysis System.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import logging
from dataclasses import dataclass

from src.core.data_models import AgentResult, ProcessingLog
from src.utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"agent.{name}")
        self.metrics = {
            "processed_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0
        }
    
    @abstractmethod
    async def process(self, data: Any) -> AgentResult:
        """
        Process the input data and return results.
        
        Args:
            data: Input data to process
            
        Returns:
            AgentResult containing the processing results
        """
        pass
    
    async def execute(self, data: Any) -> AgentResult:
        """
        Execute the agent processing with error handling and metrics collection.
        
        Args:
            data: Input data to process
            
        Returns:
            AgentResult containing the processing results
        """
        start_time = time.time()
        self.metrics["processed_count"] += 1
        
        try:
            self.logger.info(f"Starting processing for {self.name}")
            result = await self.process(data)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            if result.success:
                self.metrics["success_count"] += 1
                self.logger.info(f"Successfully processed data in {processing_time:.2f}s")
            else:
                self.metrics["error_count"] += 1
                self.logger.warning(f"Processing failed: {result.error_message}")
            
            self.metrics["total_processing_time"] += processing_time
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics["error_count"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            error_msg = f"Unexpected error in {self.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        avg_processing_time = (
            self.metrics["total_processing_time"] / max(self.metrics["processed_count"], 1)
        )
        
        success_rate = (
            self.metrics["success_count"] / max(self.metrics["processed_count"], 1)
        )
        
        return {
            "agent_name": self.name,
            "processed_count": self.metrics["processed_count"],
            "success_count": self.metrics["success_count"],
            "error_count": self.metrics["error_count"],
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.metrics["total_processing_time"]
        }
    
    def reset_metrics(self):
        """Reset agent metrics."""
        self.metrics = {
            "processed_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_processing_time": 0.0
        }
    
    def validate_input(self, data: Any, required_fields: List[str]) -> List[str]:
        """
        Validate input data has required fields.
        
        Args:
            data: Input data to validate
            required_fields: List of required field names
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not data:
            errors.append("Input data is None or empty")
            return errors
        
        if hasattr(data, '__dict__'):
            # Handle dataclass or object with attributes
            data_dict = data.__dict__
        elif isinstance(data, dict):
            # Handle dictionary
            data_dict = data
        else:
            errors.append(f"Unsupported data type: {type(data)}")
            return errors
        
        for field in required_fields:
            if field not in data_dict:
                errors.append(f"Missing required field: {field}")
            elif not data_dict[field]:
                errors.append(f"Empty value for required field: {field}")
        
        return errors
    
    def create_processing_log(
        self, 
        source_id: str, 
        action: str, 
        details: str, 
        confidence: Optional[float] = None,
        status: str = "success"
    ) -> ProcessingLog:
        """
        Create a processing log entry.
        
        Args:
            source_id: ID of the source item being processed
            action: Action being performed
            details: Additional details about the processing
            confidence: Confidence score if applicable
            status: Processing status
            
        Returns:
            ProcessingLog entry
        """
        return ProcessingLog(
            source_id=source_id,
            agent_name=self.name,
            action=action,
            details=details,
            confidence_score=confidence,
            status=status
        )
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for this agent.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def is_healthy(self) -> bool:
        """
        Check if the agent is in a healthy state.
        
        Returns:
            True if agent is healthy, False otherwise
        """
        # Basic health check - can be overridden by subclasses
        if self.metrics["processed_count"] == 0:
            return True
        
        success_rate = self.metrics["success_count"] / self.metrics["processed_count"]
        return success_rate >= 0.8  # 80% success rate threshold
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name}, config={self.config})"


class CrewAIAgentWrapper(BaseAgent):
    """
    Wrapper class for integrating with CrewAI agents.
    """
    
    def __init__(self, name: str, crew_agent: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CrewAI agent wrapper.
        
        Args:
            name: Name of the agent
            crew_agent: CrewAI agent instance
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.crew_agent = crew_agent
    
    async def process(self, data: Any) -> AgentResult:
        """
        Process data using the wrapped CrewAI agent.
        
        Args:
            data: Input data to process
            
        Returns:
            AgentResult containing the processing results
        """
        try:
            # Convert data to appropriate format for CrewAI agent
            crew_input = self._prepare_crew_input(data)
            
            # Execute CrewAI agent
            crew_result = await self.crew_agent.execute(crew_input)
            
            # Convert CrewAI result to our format
            agent_result = self._process_crew_result(crew_result)
            
            return agent_result
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"CrewAI execution failed: {str(e)}"
            )
    
    def _prepare_crew_input(self, data: Any) -> Dict[str, Any]:
        """Prepare input data for CrewAI agent."""
        # This should be implemented based on specific CrewAI agent requirements
        if hasattr(data, '__dict__'):
            return data.__dict__
        elif isinstance(data, dict):
            return data
        else:
            return {"data": data}
    
    def _process_crew_result(self, crew_result: Any) -> AgentResult:
        """Process CrewAI result and convert to AgentResult."""
        # This should be implemented based on specific CrewAI agent output format
        try:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=crew_result,
                confidence=getattr(crew_result, 'confidence', None)
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Failed to process CrewAI result: {str(e)}"
            )


# Utility functions for agent management
def create_agent_chain(agents: List[BaseAgent]) -> List[BaseAgent]:
    """
    Create a chain of agents for sequential processing.
    
    Args:
        agents: List of agents to chain
        
    Returns:
        List of agents configured for chaining
    """
    for i, agent in enumerate(agents):
        agent.config["chain_position"] = i
        agent.config["is_first"] = i == 0
        agent.config["is_last"] = i == len(agents) - 1
    
    return agents


async def execute_agent_chain(agents: List[BaseAgent], initial_data: Any) -> List[AgentResult]:
    """
    Execute a chain of agents sequentially.
    
    Args:
        agents: List of agents to execute
        initial_data: Initial input data
        
    Returns:
        List of agent results
    """
    results = []
    current_data = initial_data
    
    for agent in agents:
        result = await agent.execute(current_data)
        results.append(result)
        
        if not result.success:
            # Stop chain execution if an agent fails
            break
        
        # Pass result data to next agent
        if result.data is not None:
            current_data = result.data
    
    return results