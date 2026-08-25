import torch
import gpytorch
from pathlib import Path
import sys

# Add project root and parent to sys.path
ROOT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = ROOT_DIR.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from extensions.custom_factory.custom_anova_kernel_factory import CustomAnovaKernelFactory

def inspect_anova_params():
    model_path = ROOT_DIR / "output" / "20251229_110543 add" / "step3" / "base_gp_state.pth"
    if not model_path.exists():
        print("Model not found")
        return

    state = torch.load(model_path, map_location="cpu")
    
    # We need to know the dim_specs to reconstruct the factory
    dim_specs = [
      {"name": "x1_CeilingHeight", "type": "continuous"},
      {"name": "x2_GridModule", "type": "continuous"},
      {"name": "x3_OuterFurniture", "type": "discrete", "n_categories": 3},
      {"name": "x4_VisualBoundary", "type": "discrete", "n_categories": 3},
      {"name": "x5_PhysicalBoundary", "type": "discrete", "n_categories": 2},
      {"name": "x6_InnerFurniture", "type": "discrete", "n_categories": 3},
    ]
    
    factory = CustomAnovaKernelFactory(dim=6, dim_specs=dim_specs, interaction_mode="none")
    covar_module = factory._make_covar_module()
    
    # Load state dict (only the covar_module part)
    # The state dict in the file has 'model' and 'likelihood'
    model_state = state['model']
    
    # Extract covar_module parameters
    prefix = "covar_module."
    covar_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    
    covar_module.load_state_dict(covar_state)
    covar_module.eval()
    
    print("\nANOVA Model Parameters (Main Effects Only):")
    print("-" * 60)
    print(f"{'Dimension':<25} | {'Outputscale':<12} | {'Lengthscale':<12}")
    print("-" * 60)
    
    # In AdditiveKernel, kernels are in .kernels
    for i, sub_k in enumerate(covar_module.kernels):
        name = dim_specs[i]["name"]
        os = sub_k.outputscale.item()
        
        # base_kernel is Matern or Categorical
        base = sub_k.base_kernel
        ls = base.lengthscale.item()
        
        print(f"{name:<25} | {os:<12.4f} | {ls:<12.4f}")

if __name__ == "__main__":
    inspect_anova_params()
