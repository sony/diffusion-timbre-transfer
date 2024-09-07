import random

import torch
from torch import Tensor, nn


class RandomCrop(nn.Module):
    """Crops random chunk from the waveform"""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size
    
    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]

        if length < self.size:
            # If the tensor is smaller than the required size, it is filled with zeros
            padding_length = self.size - length
            padding = torch.zeros((x.shape[0], padding_length), dtype=x.dtype, device=x.device)
            return torch.cat([x, padding], dim=1)
        else:
            # If the tensor is larger, select a random piece
            start = random.randint(0, length - self.size)
            return x[:, start:start + self.size]


        
