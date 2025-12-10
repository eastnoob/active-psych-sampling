import torch
from aepsych.generators import ManualGenerator

# 直接测试ManualGenerator
print('=== Direct ManualGenerator Test ===')

# Test 1: With actual values
gen1 = ManualGenerator(lb=[0], ub=[2], points=[[2.8]])
print(f'Test 1: points=[[2.8]]')
print(f'  gen.points: {gen1.points}')
point1 = gen1.gen(1)
print(f'  gen.gen(1): {point1}')

# Test 2: With indices
gen2 = ManualGenerator(lb=[0], ub=[2], points=[[0]])
print(f'\nTest 2: points=[[0]]')
print(f'  gen.points: {gen2.points}')
point2 = gen2.gen(1)
print(f'  gen.gen(1): {point2}')

# Test 3: With index 1
gen3 = ManualGenerator(lb=[0], ub=[2], points=[[1]])
print(f'\nTest 3: points=[[1]]')
print(f'  gen.points: {gen3.points}')
point3 = gen3.gen(1)
print(f'  gen.gen(1): {point3}')
