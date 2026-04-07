"""
Data schemas for training dataset creation.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class TrainingExample(BaseModel):
    """Single example in the SFT dataset (veRL-compatible: prompt + response only)."""
    prompt: str = Field(..., description="Original text (user/instruction input)")
    response: str = Field(..., description="Generated latents in raw z_n: format")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for parquet export."""
        return {"prompt": self.prompt, "response": self.response}


class GenerationOutput(BaseModel):
    """Single generation output from step 5 (local model inference)."""
    generated_id: str = Field(..., description="Unique generation identifier")
    prompt: str = Field(..., description="Input prompt (original text)")
    response: str = Field(..., description="Raw generated text with z_n: labels")


class VerificationStats(BaseModel):
    """Statistics from verification process."""
    total_processed: int = 0
    passed: int = 0
    failed: int = 0
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.passed / self.total_processed) * 100
    
    def record_failure(self, reason: str):
        """Record a failure and increment its count."""
        self.failed += 1
        self.failure_breakdown[reason] = self.failure_breakdown.get(reason, 0) + 1
    
    def record_pass(self):
        """Record a successful verification."""
        self.passed += 1
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Verification Statistics:",
            f"  Total Processed: {self.total_processed}",
            f"  Passed: {self.passed}",
            f"  Failed: {self.failed}",
            f"  Pass Rate: {self.pass_rate:.2f}%",
        ]
        if self.failure_breakdown:
            lines.append(f"  Failure Breakdown:")
            for reason, count in sorted(self.failure_breakdown.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    - {reason}: {count}")
        return "\n".join(lines)
