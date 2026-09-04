from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSpecialistModel(ABC):
    """
    Abstract base class for all specialist remote sensing tools.
    Every tool must implement standard metadata properties and execution methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the specialist tool/model."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the tool/model."""
        pass

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool model.
        
        Args:
            inputs: Dictionary containing input parameters (e.g. image_path, queries, etc.)
            
        Returns:
            Dictionary containing the analysis results, confidence, and execution trace evidence.
        """
        pass
