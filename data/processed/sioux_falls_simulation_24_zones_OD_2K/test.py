import torch

data = torch.load("data/processed/sioux_falls_simulation_24_zones_OD_2K/test.pt")

data = data[0]

print(data.x)
