"""
Data schemas for batch processing, verification, and training dataset creation.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field


class LatentLayerOutput(BaseModel):
    """Single latent layer (z_n) output."""
    layer_name: str = Field(..., description="Layer identifier (e.g., 'z_4', 'z_3')")
    content: str = Field(..., description="Text content of the layer (without z_n: prefix)")
    word_count: int = Field(..., description="Number of words in the content")
    
    @property
    def layer_index(self) -> int:
        """Extract numeric index from layer name (z_4 -> 4)."""
        return int(self.layer_name.split('_')[1])


class BatchOutputItem(BaseModel):
    """Parsed batch API response item."""
    custom_id: str = Field(..., description="Unique request identifier")
    content: str = Field(..., description="Raw completion content from API")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    error: Optional[dict] = Field(None, description="Error details if request failed")
    model: Optional[str] = Field(None, description="Model used for generation")
    
    @property
    def has_error(self) -> bool:
        """Check if the batch item has an error."""
        return self.error is not None or (self.status_code and self.status_code != 200)


class VerificationResult(BaseModel):
    """Result of verification for a single batch output."""
    custom_id: str = Field(..., description="Request identifier")
    passed: bool = Field(..., description="Whether verification passed")
    raw_content: str = Field("", description="Raw content preserving z_n: format")
    layers: Optional[list[LatentLayerOutput]] = Field(None, description="Parsed layers if valid")
    failure_reasons: list[str] = Field(default_factory=list, description="Reasons for failure")
    
    def add_failure_reason(self, reason: str):
        """Add a failure reason to the list."""
        self.failure_reasons.append(reason)
        self.passed = False


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
