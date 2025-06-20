import torch  


test_tensor = torch.randn(1, device="cuda", dtype=torch.float16)
down_samples = (test_tensor,)
print(down_samples)

for i in range(3):
    down_samples += torch.randn(1, device="cuda", dtype=torch.float16),
    print(down_samples)

print(down_samples)