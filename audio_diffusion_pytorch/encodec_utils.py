import torch
from torch import Tensor
from transformers import EncodecModel
from typing import Optional

class NormalizedEncodec:
    def __init__(self, model_name: str = "facebook/encodec_24khz", device: str ='cuda'):
        self.device = device
        self.encodec_model = EncodecModel.from_pretrained(model_name).to(self.device)

    def encode_latent(self, x: Tensor, mean: Optional[Tensor], std: Optional[Tensor]) -> Tensor:
        encoder = self.encodec_model.get_encoder()
        with torch.no_grad():
            z = encoder(x)
        if mean is not None and std is not None:
            z = z_norm(z, mean, std)
        return z

    def decode_latent(self, z: Tensor, mean: Optional[Tensor], std: Optional[Tensor]) -> Tensor:
        decoder = self.encodec_model.get_decoder()
        if mean is not None and std is not None:
            z = z_denorm(z, mean, std)
        with torch.no_grad():
            x = decoder(z)
        return x

def z_norm(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (x - mean.to(x.device)) / std.to(x.device)

def z_denorm(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return x * std.to(x.device) + mean.to(x.device)
