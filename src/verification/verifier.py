"""
Base verifier interface for output validation.
"""
from abc import ABC, abstractmethod
from typing import Any

from src.data.schemas import BatchOutputItem, VerificationResult


class BaseVerifier(ABC):
    """Abstract base class for output verification."""
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize verifier with configuration.
        
        Args:
            config: Configuration dictionary containing verification rules
        """
        self.config = config
    
    @abstractmethod
    def verify(self, item: BatchOutputItem) -> VerificationResult:
        """
        Verify a single batch output item.
        
        Args:
            item: Parsed batch output to verify
        
        Returns:
            VerificationResult with pass/fail status and details
        """
        pass
    
    def verify_batch(self, items: list[BatchOutputItem]) -> list[VerificationResult]:
        """
        Verify multiple batch output items.
        
        Args:
            items: List of batch outputs to verify
        
        Returns:
            List of verification results
        """
        return [self.verify(item) for item in items]
