import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute Capability:", torch.cuda.get_device_capability(0))

# 测试张量创建
x = torch.randn(3, 3).cuda()
print("Tensor on GPU:", x)
