import sys
import os
import importlib
import inspect
import pkgutil
from pathlib import Path

# Add project root and parent to sys.path
ROOT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = ROOT_DIR.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from typing import List, Dict, Any, Tuple
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core.context import Context
from core.config_manager import ConfigManager
from core.base_module import BaseModule

# Configure Loguru
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", colorize=True)
logger.add(LOG_DIR / "warmup.log", rotation="10 MB")

console = Console()

def discover_modules() -> Dict[str, Tuple[str, BaseModule]]:
    """Dynamically discover modules in the modules/ directory."""
    modules_dict = {}
    modules_path = ROOT_DIR / "modules"
    
    # Ensure modules is a package
    if not (modules_path / "__init__.py").exists():
        with open(modules_path / "__init__.py", "w") as f:
            f.write("# Modules package\n")

    for loader, module_name, is_pkg in pkgutil.iter_modules([str(modules_path)]):
        if is_pkg:
            continue
            
        try:
            # Import the module
            full_module_name = f"modules.{module_name}"
            module = importlib.import_module(full_module_name)
            
            # Find classes that inherit from BaseModule
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseModule) and obj is not BaseModule:
                    # Extract ID from filename (e.g., step1_5 -> 1.5)
                    step_id = module_name.replace("step", "").replace("_", ".")
                    modules_dict[step_id] = (module_name, obj())
                    logger.debug(f"Discovered module: {module_name} as ID {step_id}")
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            
    # Sort by ID (numeric sort)
    sorted_keys = sorted(modules_dict.keys(), key=lambda x: [int(i) if i.isdigit() else i for i in x.split('.')])
    return {k: modules_dict[k] for k in sorted_keys}

# Discover modules at startup
MODULES = discover_modules()

def show_welcome():
    console.print(Panel.fit(
        "[bold blue]EUR-Warmup Management System[/bold blue]\n"
        "[dim]Modular CLI for Experimental Warmup Phase[/dim]",
        border_style="green"
    ))

def get_valid_steps(input_str: str) -> List[str]:
    if input_str.lower() == "all":
        return list(MODULES.keys())
    
    steps = [s.strip() for s in input_str.split(",")]
    valid_steps = []
    for s in steps:
        if s in MODULES:
            valid_steps.append(s)
        else:
            logger.warning(f"Unknown step: {s}")
    return valid_steps

def check_logic(steps: List[str]) -> bool:
    # Simple logic check: Step 2 needs Step 1.5 or previous data
    if "2" in steps and "1" in steps and "1.5" not in steps:
        logger.error("Invalid combination: Step 2 requires Step 1.5 (Simulation) if running with Step 1.")
        return False
    return True

def handle_step2_review(context: Context, config_manager: ConfigManager):
    """Pause and allow user to review/edit the model spec after Step 2."""
    if not hasattr(context, "model_spec_path") or not context.model_spec_path:
        return

    spec_path = Path(context.model_spec_path)
    if not spec_path.exists():
        return

    console.print(Panel(
        f"[bold yellow]Step 2 Analysis Complete![/bold yellow]\n\n"
        f"Analyzer generated a model recipe: [cyan]{spec_path.name}[/cyan]\n"
        f"Path: [dim]{spec_path}[/dim]\n\n"
        "Please review the files. You can manually adjust interactions or priors.\n"
        "Save changes and press [bold green]Enter[/bold green] to continue to Step 3, or type [bold red]'q'[/bold red] to quit.",
        title="Human-in-the-loop Review",
        border_style="yellow"
    ))

    # Auto-open editor (JSON and Summary)
    if hasattr(context, "model_summary_path") and Path(context.model_summary_path).exists():
        config_manager.open_editor(Path(context.model_summary_path))
    
    config_manager.open_editor(spec_path)

    choice = Prompt.ask("Confirm config and continue?", default="")
    if choice.lower() == 'q':
        raise InterruptedError("User terminated the workflow after Step 2 review.")

def main():
    show_welcome()
    
    config_manager = ConfigManager(Path("config"))
    
    # --- Context Initialization Logic ---
    output_root = Path("output")
    existing_runs = []
    if output_root.exists():
        existing_runs = sorted([d.name for d in output_root.iterdir() if d.is_dir()], reverse=True)
    
    context = Context()
    
    if existing_runs:
        console.print("\n[bold cyan]Workflow Context Selection:[/bold cyan]")
        console.print(f"  [green]n[/green]: Start completely NEW run (Timestamp: {context.timestamp}) [bold](default)[/bold]")
        for i, run in enumerate(existing_runs[:5]):
            console.print(f"  [yellow]{i}[/yellow]: Resume/Use previous run [dim]({run})[/dim]")
        console.print("  [yellow]m[/yellow]: Manual input directory name")
        
        choice = Prompt.ask("\nSelect context (n/ID/m)", default="n")
        
        selected_run = None
        if choice.lower() == 'n':
            logger.info(f"Starting new workflow run: {context.timestamp}")
        elif choice.lower() == 'm':
            selected_run = Prompt.ask("Enter directory name (e.g., 20231225_120000)")
        elif choice.isdigit() and int(choice) < len(existing_runs):
            selected_run = existing_runs[int(choice)]
        
        if selected_run:
            run_path = output_root / selected_run
            if not run_path.exists():
                console.print(f"[red]Error: Directory {selected_run} does not exist.[/red]")
            else:
                context.timestamp = selected_run
                logger.info(f"Resuming context: {selected_run}")
                
                # 1. Recover subject files
                # 逻辑：回溯时，我们需要找回该目录下已有的所有 CSV 文件
                # 优先找 step1_5 (模拟后的数据)，如果没有则找 step1 (原始采样方案)
                s1_dir = run_path / "step1"
                s1_5_dir = run_path / "step1_5"
                
                files = []
                if s1_5_dir.exists():
                    # 只找 subject_*.csv，避免包含 combined_results.csv
                    files = [str(f) for f in s1_5_dir.glob("subject_*.csv")]
                    if files:
                        logger.info(f"Recovered {len(files)} simulated files from step1_5")
                
                if not files and s1_dir.exists():
                    files = [str(f) for f in s1_dir.glob("subject_*.csv")]
                    if files:
                        logger.info(f"Recovered {len(files)} base sampling files from step1")
                
                context.subject_files = files
                
                # 2. Recover analysis results (Step 2)
                s2_dir = run_path / "step2"
                s2_results = s2_dir / "analysis_results.json"
                if s2_results.exists():
                    import json
                    with open(s2_results, 'r') as f:
                        context.analysis_results = json.load(f)
                    logger.info("Recovered Step 2 analysis results")
                
                s2_spec = s2_dir / "model_spec.json"
                if s2_spec.exists():
                    context.model_spec_path = str(s2_spec)
                    logger.info(f"Recovered model spec path: {s2_spec}")
                
                s2_summary = s2_dir / "model_spec_summary.md"
                if s2_summary.exists():
                    context.model_summary_path = str(s2_summary)
        else:
            # 默认新建
            logger.info(f"New run directory: output/{context.timestamp}")
    # ------------------------------------
    
    while True:
        console.print("\n[bold]Available Steps:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim")
        table.add_column("Name")
        table.add_column("Description")
        
        for step_id, (mod_name, mod_inst) in MODULES.items():
            name = getattr(mod_inst, "name", mod_name)
            desc = getattr(mod_inst, "description", mod_inst.__doc__ or "")
            table.add_row(step_id, name, desc)
            
        table.add_row("all", "Full Flow", f"Run {' -> '.join(MODULES.keys())}")
        
        console.print(table)
        
        cmd = Prompt.ask("\nEnter steps to run (e.g., '1,1.5,2' or 'all', 'q' to quit)")
        
        if cmd.lower() == 'q':
            break
            
        selected_ids = get_valid_steps(cmd)
        if not selected_ids:
            continue
            
        if not check_logic(selected_ids):
            continue
            
        # Prepare config
        modules_to_run = [MODULES[sid] for sid in selected_ids]
        
        existing_configs = config_manager.list_configs()
        config_choice = "n"
        if existing_configs:
            console.print("\n[bold]Configuration Selection:[/bold]")
            console.print("n: Create new configuration from templates")
            for i, cfg in enumerate(existing_configs):
                console.print(f"{i}: Use existing {cfg.name}")
            
            config_choice = Prompt.ask("Select configuration option", default="n")

        if config_choice == "n":
            save_name = Prompt.ask("Save new config as (e.g. 'my_run.toml', leave blank for temp)", default="")
            filename = save_name if save_name.endswith(".toml") else (f"{save_name}.toml" if save_name else "temp_config.toml")
            
            config_info = [(name, mod.get_default_config()) for name, mod in modules_to_run]
            config_path = config_manager.create_combined_config(config_info, filename=filename)
            console.print(f"\n[yellow]Config file generated at: {config_path}[/yellow]")
            config_manager.open_editor(config_path)
        else:
            try:
                idx = int(config_choice)
                config_path = existing_configs[idx]
                if Prompt.ask(f"Edit {config_path.name} before running?", choices=["y", "n"], default="n") == "y":
                    config_manager.open_editor(config_path)
            except (ValueError, IndexError):
                logger.error("Invalid selection, defaulting to new config.")
                config_info = [(name, mod.get_default_config()) for name, mod in modules_to_run]
                config_path = config_manager.create_combined_config(config_info)
                config_manager.open_editor(config_path)

        confirm = Prompt.ask("Config ready? Press Enter to validate and run ('c' to cancel)", default="")
        if confirm.lower() == 'c':
            continue
            
        # Load and validate
        try:
            config = config_manager.load_and_validate(config_path)
            
            # Validate each module
            all_valid = True
            for name, mod in modules_to_run:
                valid, errors = mod.validate(config)
                if not valid:
                    all_valid = False
                    for err in errors:
                        logger.error(f"[{name}] {err}")
            
            if not all_valid:
                console.print("[red]Configuration validation failed. Please fix the errors and try again.[/red]")
                continue
                
            # Confirm execution
            if not Prompt.ask("Configuration valid. Start execution?", choices=["y", "n"], default="y") == "y":
                continue
                
            # Run modules
            for i, (name, mod) in enumerate(modules_to_run):
                logger.info(f">>> Starting {name}...")
                context = mod.run(config, context)
                logger.success(f"<<< {name} finished.")
                
                # Check if we need to pause after Step 2
                if name == "step2" and i < len(modules_to_run) - 1 and modules_to_run[i+1][0] == "step3":
                    handle_step2_review(context, config_manager)
                
            console.print("\n[bold green]All selected steps completed successfully![/bold green]")
            
        except InterruptedError as e:
            logger.warning(str(e))
        except Exception as e:
            logger.exception(f"An error occurred during execution: {e}")
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
