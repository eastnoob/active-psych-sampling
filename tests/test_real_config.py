from aepsych.config import Config
from aepsych.transforms.parameters import ParameterTransforms

config = Config()
config.update(config_fnames=['d:/ENVS/active-psych-sampling/tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini'])

# Get transforms
transforms = ParameterTransforms.from_config(config)

print('=== Transforms Created ===')
for name, transform in transforms._modules.items():
    print(f'{name}: {type(transform).__name__}')
    
print(f'\n=== Test x1 (categorical numeric) ===')
import torch
test_input = torch.tensor([[2.8, 6.5, 0, 0, 0, 0]])
print(f'Input: {test_input[0,:2]}')
transformed = transforms.transform(test_input)
print(f'After transform: {transformed[0,:2]}')
untransformed = transforms.untransform(transformed)
print(f'After untransform: {untransformed[0,:2]}')
