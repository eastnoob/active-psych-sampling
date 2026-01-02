#!/usr/bin/env python3
"""EUR Experiment Runner - Main CLI entry point."""

import sys
from pathlib import Path

# Add project root and parent project to path
ROOT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = ROOT_DIR.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from core.context import Context
from core.config_manager import ConfigManager
from utils.oracle import LinearOracle
from roles.sobol_role import SobolRole
from roles.random_role import RandomRole
from roles.eur_role import EURRole
from behaviors.single_run import SingleRun
from behaviors.multi_subject_run import MultiSubjectRun

# Import custom components for KEY_CESHI replication
try:
    from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator
    from extensions.dynamic_eur_acquisition.eur_anova_multi import EURAnovaMultiAcqf
    from extensions.custom_factory import CustomBaseGPResidualFactory
    logger.info("Custom components imported successfully")
except ImportError as e:
    logger.warning(f"Custom components not available: {e}")

# Configure loguru
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add(LOG_DIR / "experiment.log", rotation="10 MB", level="DEBUG")

console = Console()


def show_welcome():
    """Display welcome message."""
    console.print(Panel.fit(
        "[bold blue]EUR Experiment Runner[/bold blue]\n"
        "[dim]Modular framework for testing acquisition methods[/dim]",
        border_style="green"
    ))


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Path to TOML configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(config: str, verbose: bool):
    """EUR Experiment Runner - Test and compare acquisition methods.

    Example:
        python main.py --config config/sobol_test.toml
    """
    show_welcome()

    if verbose:
        logger.remove()
        logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="DEBUG")
        logger.add(LOG_DIR / "experiment.log", rotation="10 MB", level="DEBUG")

    try:
        # Load configuration
        config_manager = ConfigManager(Path(config))
        cfg = config_manager.load()

        console.print(f"\n[bold]Configuration loaded:[/bold] {config}")

        # Create context
        context = Context()

        # Setup output directory
        output_dir = Path(cfg.get('experiment', {}).get('output_dir', 'output/default'))
        output_dir.mkdir(parents=True, exist_ok=True)
        context.output_dir = output_dir

        # Create oracle
        oracle_cfg = cfg.get('oracle', {})
        oracle_type = oracle_cfg.get('type', 'linear')

        if oracle_type == 'linear':
            linear_cfg = oracle_cfg.get('linear', {})
            # Parse interactions from config
            interactions_cfg = linear_cfg.get('interactions', {})
            interactions = {}
            for key, value in interactions_cfg.items():
                # Parse "i,j" format to (i, j) tuple
                i, j = map(int, key.split(','))
                interactions[(i, j)] = float(value)

            oracle = LinearOracle(
                seed=oracle_cfg.get('seed', 42),
                noise_std=oracle_cfg.get('noise_std', 0.5),
                output_type=oracle_cfg.get('output_type', 'binary'),
                likert_levels=oracle_cfg.get('likert_levels', 5),
                weights=linear_cfg.get('weights'),
                bias=linear_cfg.get('bias', 0.0),
                interactions=interactions
            )
            context.oracle = oracle
            console.print(f"[green]OK[/green] Oracle created: {oracle_type}")
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")

        # Create role
        role_cfg = cfg.get('role', {})
        role_type = role_cfg.get('type', 'sobol')

        if role_type == 'sobol':
            role = SobolRole()
            console.print(f"[green]OK[/green] Role created: {role_type}")
        elif role_type == 'random':
            role = RandomRole()
            console.print(f"[green]OK[/green] Role created: {role_type}")
        elif role_type == 'eur':
            role = EURRole()
            console.print(f"[green]OK[/green] Role created: {role_type}")
        else:
            raise ValueError(f"Unknown role type: {role_type}")

        # Create behavior
        behavior_cfg = cfg.get('behavior', {})
        behavior_type = behavior_cfg.get('type', 'single_run')

        if behavior_type == 'single_run':
            behavior = SingleRun()
            console.print(f"[green]OK[/green] Behavior created: {behavior_type}")

            # Validate behavior config
            behavior_params = behavior_cfg.get(behavior_type, {})
            is_valid, errors = behavior.validate(behavior_params)
            if not is_valid:
                console.print(f"[red]ERROR[/red] Configuration errors:")
                for error in errors:
                    console.print(f"  - {error}")
                sys.exit(1)

            # Run experiment
            console.print("\n[bold]Starting experiment...[/bold]\n")
            behavior.run([role], cfg, context)

            # Show results
            console.print(f"\n[bold green]OK Experiment completed successfully![/bold green]")
            console.print(f"Results saved to: [cyan]{context.output_dir}[/cyan]")

            # Show summary table
            summary_data = context.get_result(role.get_name())
            if summary_data:
                history = summary_data.get('history', [])
                total_trials = len(history)
                warmup_trials = len([h for h in history if h.get('phase') == 'warmup'])
                main_trials = total_trials - warmup_trials

                table = Table(title="Experiment Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Total Trials", str(total_trials))
                table.add_row("Warmup Trials", str(warmup_trials))
                table.add_row("Main Trials", str(main_trials))
                table.add_row("Output Directory", str(context.output_dir))

                console.print("\n")
                console.print(table)

        elif behavior_type == 'multi_subject_run':
            behavior = MultiSubjectRun()
            console.print(f"[green]OK[/green] Behavior created: {behavior_type}")

            # Validate behavior config
            behavior_params = behavior_cfg.get(behavior_type, {})
            is_valid, errors = behavior.validate(behavior_params)
            if not is_valid:
                console.print(f"[red]ERROR[/red] Configuration errors:")
                for error in errors:
                    console.print(f"  - {error}")
                sys.exit(1)

            # Run experiment
            console.print("\n[bold]Starting multi-subject experiment...[/bold]\n")
            behavior.run([role], behavior_params, context)

            # Show results
            console.print(f"\n[bold green]OK Multi-subject experiment completed successfully![/bold green]")
            console.print(f"Results saved to: [cyan]{context.output_dir}[/cyan]")

            # Show summary table
            aggregate_path = context.output_dir / "aggregate_summary.json"
            if aggregate_path.exists():
                import json
                with open(aggregate_path, 'r') as f:
                    aggregate = json.load(f)

                table = Table(title="Multi-Subject Experiment Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Number of Subjects", str(aggregate['n_subjects']))
                table.add_row("Budget per Subject", str(aggregate['budget_per_subject']))
                table.add_row("Warmup per Subject", str(aggregate['warmup_per_subject']))
                table.add_row("Total Trials", str(aggregate['total_trials']))
                table.add_row("Output Directory", str(context.output_dir))

                console.print("\n")
                console.print(table)

        else:
            raise ValueError(f"Unknown behavior type: {behavior_type}")

    except Exception as e:
        logger.exception("Experiment failed")
        console.print(f"\n[bold red]ERROR Experiment failed:[/bold red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
