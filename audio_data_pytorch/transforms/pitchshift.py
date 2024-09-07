import torchaudio
from torch import Tensor, nn
import random
import torch
import numpy as np

class PitchShift(nn.Module):
    """Shift Audio for a the num of semitones"""

    def __init__(self, sign):
        super().__init__()
        self.sign = sign
        self.trans_pool = [torchaudio.transforms.PitchShift(sample_rate=24000, n_steps=i*sign) for i in np.arange(1,26)]
    def forward(self, x: Tensor) -> Tensor:
        # Apply only with 35% chance
        if random.random() < 0.35:
            transform = np.random.choice(self.trans_pool)
            with torch.no_grad():
                x = transform(x)
        return x
