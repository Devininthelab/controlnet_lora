from diffusion.controlnet import ControlNetConditioningEmbedding
import torch

model = ControlNetConditioningEmbedding(
    conditioning_embedding_channels=64,
    conditioning_channels=3).to(device='cuda')

test_tensor = torch.randn(1, 3, 512, 512).to(device='cuda')
print(model(test_tensor).shape)  