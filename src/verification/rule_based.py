"""
Rule-based verification for latent layer outputs.
"""
import re
from typing import Any

from src.data.schemas import BatchOutputItem, LatentLayerOutput, VerificationResult
from src.verification.verifier import BaseVerifier


class RuleBasedVerifier(BaseVerifier):
    """Rule-based verifier for coarse-to-fine latent layer outputs."""
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize with configuration containing word count constraints.
        
        Args:
            config: Configuration dict with 'word_count_constraints' key
        """
        super().__init__(config)
        self.word_count_constraints = config.get('word_count_constraints', {})
        self.strict_word_count = config.get('verification', {}).get('strict_word_count', True)
        self.check_layer_order = config.get('verification', {}).get('check_layer_order', True)
        self.allow_extra_whitespace = config.get('verification', {}).get('allow_extra_whitespace', True)
        
        # Expected layer order (coarse to fine)
        self.expected_layers = ['z_4', 'z_3', 'z_2', 'z_1']
    
    def verify(self, item: BatchOutputItem) -> VerificationResult:
        """
        Verify a single batch output against rules.
        
        Args:
            item: Parsed batch output to verify
        
        Returns:
            VerificationResult with pass/fail status and reasons
        """
        result = VerificationResult(custom_id=item.custom_id, passed=True, raw_content=item.content)

        # Rule 1: Check for API errors
        if item.has_error:
            result.add_failure_reason(f"API error: {item.error or f'status_code={item.status_code}'}")
            return result
        
        # Rule 2: Parse layers from content
        layers = self._parse_layers(item.content)
        if layers is None:
            result.add_failure_reason("Failed to parse layer structure")
            return result
        
        # Rule 3: Check layer count
        if len(layers) != len(self.expected_layers):
            result.add_failure_reason(
                f"Expected {len(self.expected_layers)} layers, found {len(layers)}"
            )
            return result
        
        # Rule 4: Check layer order
        if self.check_layer_order:
            layer_names = [layer.layer_name for layer in layers]
            if layer_names != self.expected_layers:
                result.add_failure_reason(
                    f"Layer order mismatch: expected {self.expected_layers}, got {layer_names}"
                )
                return result
        
        # Rule 5: Check word counts
        for layer in layers:
            expected_count = self.word_count_constraints.get(layer.layer_name)
            if expected_count is None:
                result.add_failure_reason(f"No word count constraint for {layer.layer_name}")
                return result
            
            if self.strict_word_count and layer.word_count != expected_count:
                result.add_failure_reason(
                    f"{layer.layer_name}: expected {expected_count} words, got {layer.word_count}"
                )
                return result
        
        # Rule 6: Check for empty layers
        for layer in layers:
            if not layer.content.strip():
                result.add_failure_reason(f"{layer.layer_name} has empty content")
                return result
        
        # All checks passed
        result.layers = layers
        return result
    
    def _parse_layers(self, content: str) -> list[LatentLayerOutput] | None:
        """
        Parse latent layers from content string.
        
        Args:
            content: Raw content from API response
        
        Returns:
            List of parsed layers, or None if parsing failed
        """
        try:
            layers = []
            
            # Split content by lines
            lines = content.strip().split('\n')
            
            # Pattern to match layer lines: z_N: content
            layer_pattern = re.compile(r'^(z_\d+):\s*(.*)$')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = layer_pattern.match(line)
                if not match:
                    # Line doesn't match expected format
                    return None
                
                layer_name = match.group(1)
                layer_content = match.group(2).strip()
                
                # Count words (split on whitespace)
                if self.allow_extra_whitespace:
                    words = layer_content.split()
                else:
                    words = layer_content.split(' ')
                
                word_count = len([w for w in words if w])  # Filter empty strings
                
                layers.append(LatentLayerOutput(
                    layer_name=layer_name,
                    content=layer_content,
                    word_count=word_count
                ))
            
            return layers if layers else None
            
        except Exception as e:
            # Parsing error
            return None
