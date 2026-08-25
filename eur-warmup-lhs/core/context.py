from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

@dataclass
class Context:
    """Shared state passed between warmup modules."""
    
    # Input/Output paths
    design_space_path: Optional[str] = None
    output_root: Path = Path("output")
    
    # Data objects
    design_df: Optional[pd.DataFrame] = None
    subject_files: List[str] = field(default_factory=list)
    
    # Results from steps
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    
    def get_step_output_dir(self, step_name: str) -> Path:
        """Get a dedicated output directory for a specific step."""
        path = self.output_root / self.timestamp / step_name
        path.mkdir(parents=True, exist_ok=True)
        return path
