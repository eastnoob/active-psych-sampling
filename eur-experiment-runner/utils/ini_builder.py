"""INI configuration builder utilities."""

from typing import Dict, Any, List
from pathlib import Path
from loguru import logger


class INIBuilder:
    """Helper class for building AEPsych INI configuration files."""

    def __init__(self):
        """Initialize INI builder."""
        self.sections: List[str] = []

    def add_common_section(self, parnames: List[str], outcome_types: List[str],
                          strategy_names: List[str], lb: List[float] = None, ub: List[float] = None) -> 'INIBuilder':
        """Add [common] section.

        Args:
            parnames: Parameter names
            outcome_types: Outcome types (e.g., ['binary'])
            strategy_names: Strategy names (e.g., ['init_strat', 'eur_strat'])
            lb: Lower bounds for parameters (optional)
            ub: Upper bounds for parameters (optional)

        Returns:
            Self for chaining
        """
        section = "[common]\n"
        # Format parnames with quotes: ['x1', 'x2', 'x3']
        parnames_str = "[" + ", ".join(f"'{p}'" for p in parnames) + "]"
        section += f"parnames = {parnames_str}\n"
        section += "stimuli_per_trial = 1\n"
        # Format outcome_types without quotes: [binary]
        outcome_str = "[" + ", ".join(outcome_types) + "]"
        section += f"outcome_types = {outcome_str}\n"
        # Format strategy_names without quotes: [sobol_strat]
        strategy_str = "[" + ", ".join(strategy_names) + "]"
        section += f"strategy_names = {strategy_str}\n"

        # Add lb/ub if provided
        if lb is not None:
            lb_str = "[" + ", ".join(str(x) for x in lb) + "]"
            section += f"lb = {lb_str}\n"
        if ub is not None:
            ub_str = "[" + ", ".join(str(x) for x in ub) + "]"
            section += f"ub = {ub_str}\n"

        self.sections.append(section)
        logger.debug(f"Added [common] section with {len(parnames)} parameters")
        return self

    def add_parameter_section(self, name: str, par_type: str, **kwargs) -> 'INIBuilder':
        """Add parameter definition section.

        Args:
            name: Parameter name
            par_type: Parameter type ('continuous', 'custom_ordinal_mono', 'categorical')
            **kwargs: Additional parameter properties (lower_bound, upper_bound, values, choices, etc.)

        Returns:
            Self for chaining
        """
        section = f"[{name}]\n"
        section += f"par_type = {par_type}\n"

        if par_type == 'continuous':
            section += f"lower_bound = {kwargs.get('lower_bound', 0.0)}\n"
            section += f"upper_bound = {kwargs.get('upper_bound', 1.0)}\n"
        elif par_type == 'custom_ordinal_mono':
            values = kwargs.get('values', [])
            values_str = "[" + ", ".join(str(v) for v in values) + "]"
            section += f"values = {values_str}\n"
        elif par_type == 'categorical':
            choices = kwargs.get('choices', [])
            choices_str = "[" + ", ".join(f"'{c}'" for c in choices) + "]"
            section += f"choices = {choices_str}\n"

        self.sections.append(section)
        logger.debug(f"Added parameter section: {name} ({par_type})")
        return self

    def add_raw_section(self, section_text: str) -> 'INIBuilder':
        """Add raw INI section text.

        Args:
            section_text: Raw INI text

        Returns:
            Self for chaining
        """
        self.sections.append(section_text)
        return self

    def build(self) -> str:
        """Build complete INI configuration string.

        Returns:
            Complete INI configuration
        """
        ini_content = "\n".join(self.sections)
        logger.debug(f"Built INI config with {len(self.sections)} sections")
        return ini_content

    def save(self, path: Path):
        """Save INI configuration to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ini_content = self.build()
        with open(path, 'w') as f:
            f.write(ini_content)

        logger.info(f"Saved INI config to: {path}")
