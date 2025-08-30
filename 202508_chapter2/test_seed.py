import torch
import numpy as np

print("=== 实验1: 相同种子产生相同结果 ===")
# 第一次运行
torch.manual_seed(42)
print("第一次设置种子42:")
print(f"  torch.rand(3): {torch.rand(3)}")
print(f"  torch.randn(2,2):\n{torch.randn(2,2)}")

# 第二次运行 - 重新设置相同的种子
torch.manual_seed(42)
print("\n第二次设置种子42:")
print(f"  torch.rand(3): {torch.rand(3)}")
print(f"  torch.randn(2,2):\n{torch.randn(2,2)}")

print("\n=== 实验2: 不同种子产生不同结果 ===")
torch.manual_seed(100)
print("设置种子100:")
print(f"  torch.rand(3): {torch.rand(3)}")

print("\n=== 实验3: 种子只影响后续操作 ===")
torch.manual_seed(42)
rand1 = torch.rand(2)
rand2 = torch.rand(2)  # 这个会不同于rand1
print(f"设置种子42后:")
print(f"  第1次 torch.rand(2): {rand1}")
print(f"  第2次 torch.rand(2): {rand2}")

# 重新设置种子
torch.manual_seed(42)
rand3 = torch.rand(2)
rand4 = torch.rand(2)
print(f"\n重新设置种子42后:")
print(f"  第1次 torch.rand(2): {rand3}")  # 会等于rand1
print(f"  第2次 torch.rand(2): {rand4}")  # 会等于rand2

print("\n=== 实验4: NumPy 同样的行为 ===")
np.random.seed(42)
print(f"第一次设置NumPy种子42: {np.random.rand(3)}")
np.random.seed(42)
print(f"第二次设置NumPy种子42: {np.random.rand(3)}")