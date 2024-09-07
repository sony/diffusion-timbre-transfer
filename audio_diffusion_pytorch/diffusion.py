from math import atan, cos, pi, sin, sqrt
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
import torchaudio
import csv

import numpy as np
from tqdm import tqdm
import einops
import ot
from functools import partial

import torchaudio
import matplotlib.pyplot as plt
import librosa

from .utils import default, exists

"""
Diffusion Training
"""

""" Distributions """


class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()


class UniformDistribution(Distribution):
    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        return torch.rand(num_samples, device=device)

class KDistribution(Distribution):
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        rho: float,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        
        a = torch.rand(num_samples, device=device)
        t = (self.sigma_max**(1/self.rho) + a *(self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho)))**self.rho
        return t

class VKDistribution(Distribution):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = float("inf"),
        sigma_data: float = 1.0,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma_data = sigma_data

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        sigma_data = self.sigma_data
        min_cdf = atan(self.min_value / sigma_data) * 2 / pi
        max_cdf = atan(self.max_value / sigma_data) * 2 / pi
        u = (max_cdf - min_cdf) * torch.randn((num_samples,), device=device) + min_cdf
        return torch.tan(u * pi / 2) * sigma_data


""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs


class Diffusion(nn.Module):

    alias: str = ""

    """Base diffusion class"""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError("Diffusion class missing denoise_fn")

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        raise NotImplementedError("Diffusion class missing forward function")


class VDiffusion(Diffusion):

    alias = "v"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution, latent: bool):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")


        # Get noise
        noise = default(noise, lambda: torch.randn_like(x))

        # Combine input and noise weighted by half-circle
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta

        # Denoise and return loss
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        
        #losses = F.mse_loss(x_denoised, x_target, reduction="none") 
        #losses = reduce(losses, "b ... -> b", "mean") 
        
        return F.mse_loss(x_denoised, x_target), None, sigmas

def find_closest_divisor(num_chunks, original_batch_size):
    closest_divisor = None
    min_diff = float('inf')
    for i in range(1, num_chunks + 1):
        if num_chunks % i == 0:
            diff = abs(i - original_batch_size)
            if diff < min_diff:
                min_diff = diff
                closest_divisor = i
    return closest_divisor

def OT_plan(x_chunk, z, batch_size=1024*6):

    # Define the OT function locally
    ot_fn = partial(ot.sinkhorn, reg=0.001, numItermax=int(1e2), method='sinkhorn_log')
    algorithm = "sinkhorn"
    device = x_chunk.device
    # Compute pairwise L2 distance matrix in smaller batches to save memory
    num_chunks = x_chunk.shape[0]
    batch_size = find_closest_divisor(num_chunks, batch_size)
    M = torch.zeros((num_chunks, num_chunks), device='cpu')  # Use CPU to save GPU memory

    for i in range(0, num_chunks, batch_size):
        end_i = i + batch_size
        x_batch = x_chunk[i:end_i].unsqueeze(1).to(device)  # Move to GPU

        for j in range(0, num_chunks, batch_size):
            end_j = j + batch_size
            z_batch = z[j:end_j].unsqueeze(0).to(device)  # Move to GPU

            # Perform the operation on GPU
            result_gpu = ((x_batch - z_batch) ** 2).mean(-1).mean(-1)

            # Move the result to CPU and assign to M
            M[i:end_i, j:end_j] = result_gpu.to('cpu')

    M = M.T
    M += 1e-5  # for numerical stability

    # Assert that M is a square matrix
    assert M.shape[0] == M.shape[1]

    a, b = ot.unif(M.shape[0]), ot.unif(M.shape[1])
    a = torch.from_numpy(a).float().to(M.device)
    b = torch.from_numpy(b).float().to(M.device)

    # Apply the solver
    P = ot_fn(a, b, M)

    if algorithm == "emd":
        # In the case of deterministic OT, this is equivalent to what we do below, and probably faster
        index = P.max(axis=1).indices
    else:
        P *= x_chunk.shape[0]
        P /= P.sum(-1)
        normalized_P = P / P.sum(dim=1, keepdim=True)
        index = torch.multinomial(normalized_P, 1, replacement=True).squeeze()

    return index

def get_audio_noise_pairs(x, x_plan):
    B, C, T = x.shape

    chunk_size_t = 4
    chunk_size_c = 4
    t = T // chunk_size_t
    c = C // chunk_size_c

    # Divide the audio in chunks, handling the channel and time dimensions
    x_chunk = einops.rearrange(x_plan, "b (c c_chunk) (t t_chunk) -> (b t c)  c_chunk t_chunk", c_chunk=chunk_size_c, t_chunk=chunk_size_t)

    z = torch.randn((x_chunk.shape[0], chunk_size_c, chunk_size_t), device=x.device)

    index = OT_plan(x_chunk, z)
    z = z[index]

    z = einops.rearrange(z, "(b t c) c_chunk t_chunk -> b (c c_chunk) (t t_chunk)", b=B, c_chunk=chunk_size_c, t_chunk=chunk_size_t, c=c, t=t)

    return x, z


class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
        latent: bool
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold
        self.latent = latent

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)

        x_denoised = c_skip * x_noisy + c_out * x_pred

        if self.latent == True:
            return x_denoised 

        elif self.latent == False:
            # Clips in [-1,1] range, with dynamic thresholding if provided
            return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)

        else:
            print("Please choose latent between True of False")

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        # extract useful methods from XDiffusion
        diff_self = kwargs.get('diff_self', None)
        pl_self = kwargs.get('pl_self', None)

        #x, noise = x
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device) #sigmas = torch.tensor([1e-1, 1, 10, 100], device=device) 
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        #mean = -0.56
        #std = 5.23
        #x = (x - mean) / (std) 

        # Add noise to input
        #x_plan = x
        #x, noise = get_audio_noise_pairs(x, x_plan)
        #noise = noise.to(x.device)
        noise = default(noise, lambda: torch.randn_like(x)) 

        x_noisy = x + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)      
        
        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none") 
        losses = reduce(losses, "b ... -> b", "mean") 
        #loss_weight = self.loss_weight(sigmas) #.unsqueeze(1).unsqueeze(2)
        #loss_weight_padded = rearrange(loss_weight, "b -> b 1 1")
        #losses_before = losses
        #print("LOSSES BEFORE:", losses)
        losses = losses * self.loss_weight(sigmas)
       # losses = losses * loss_weight_padded
        #print("LOSSES:", losses)
        #print("SIGMAS:", sigmas)
        
        loss = losses.mean()
        #print("LOSS", loss)


        if pl_self.log_next: # log only for the first batch
            if pl_self.training:
                status = "Train"
            else:
                status = "Val"

            if pl_self.latent:
                decoder = pl_self.encodec_model.get_decoder()
                # obtain waveform from latent
                x_noisy = decoder(x_noisy).float()
                x_denoised = decoder(x_denoised).float()
            
            diff_self.log_tensorboard_audio(
                writer=pl_self.logger.experiment,
                id=f"{status}/x_noisy",
                samples=x_noisy,
                sampling_rate=pl_self.sampling_rate,
                step=pl_self.global_step
            )
            diff_self.log_tensorboard_spectrogram(
                writer=pl_self.logger.experiment,
                id=f"{status}/x_noisy",
                samples=x_noisy,
                sampling_rate=pl_self.sampling_rate,
                step=pl_self.global_step,
                sigmas=sigmas
            )

            diff_self.log_tensorboard_audio(
                writer=pl_self.logger.experiment,
                id=f"{status}/x_denoised",
                samples=x_denoised,
                sampling_rate=pl_self.sampling_rate,
                step=pl_self.global_step
            )

            diff_self.log_tensorboard_spectrogram(
                writer=pl_self.logger.experiment,
                id=f"{status}/x_denoised",
                samples=x_denoised,
                sampling_rate=pl_self.sampling_rate,
                step=pl_self.global_step
            )              


        return loss, losses, sigmas




class VKDiffusion(Diffusion):

    alias = "vk"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = 1.0
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = -sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        return (t * pi / 2).tan()

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        #file_path = f"/home/demancum/Samples/denoise_violin_{i}.wav"
        #torchaudio.save(file_path, x[0].cpu(), 16000)

        # Compute model output
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-7)

        # Compute loss
        loss = F.mse_loss(x_pred, v_target)
        return loss


"""
Diffusion Sampling
"""

""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: int):
        super().__init__()
        self.start = start
    def forward(self, num_steps: int, device: Any) -> Tensor:
        sigmas = torch.linspace(self.start, 0, num_steps + 1)[:-1]
        return sigmas


class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 100, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (self.sigma_max ** rho_inv + (steps / (num_steps - 1)) * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)

        return sigmas


""" Samplers """


class Sampler(nn.Module):

    diffusion_types: List[Type[Diffusion]] = []

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        raise NotImplementedError()

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        raise NotImplementedError("Inpainting not available with current sampler")


class VSampler(Sampler):

    diffusion_types = [VDiffusion]

    def get_alpha_beta(self, sigma: float) -> Tuple[float, float]:
        angle = sigma * pi / 2
        alpha = cos(angle)
        beta = sin(angle)
        return alpha, beta

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise 
        
        alpha, beta = self.get_alpha_beta(sigmas[0].item())

        for i in range(num_steps - 1):
            #print("std of x aaaaaaaaaaaoooooooooooooooo", x.std().mean().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            
            #print("sigma", sigmas[i].item())
            is_last = i == num_steps - 1

            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha

            if not is_last:
                alpha, beta = self.get_alpha_beta(sigmas[i + 1].item())
                x = x_pred * alpha + x_eps * beta
            print("std of x aaaaaaaaaaaooo", x_pred.std())
            

        return x_pred

class VSamplerReverseTMP(VSampler):
    def forward(
        self, signal: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = signal  # Starting with a clean signal
        alpha, beta = self.get_alpha_beta(sigmas[-1].item())  # Start from the lowest noise level

        for i in reversed(range(num_steps - 1)):
            is_first = i == 0

            x_denoised = fn(x, sigma=sigmas[i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha

            if not is_first:
                alpha, beta = self.get_alpha_beta(sigmas[i - 1].item())
                x = x_pred * alpha + x_eps * beta

        return x_pred

class VSamplerReverse(Sampler):

    diffusion_types = [VDiffusion]

    def get_alpha_beta(self, sigma: float) -> Tuple[float, float]:
        angle = sigma * pi / 2
        alpha = cos(angle)
        beta = sin(angle)
        return alpha, beta

    def forward(
        self, x: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        #x = sigmas[0] * noise
        alpha, beta = self.get_alpha_beta(sigmas[-1].item())

        for i in range(num_steps - 1):
            is_last = i == num_steps - 1

            x_denoised = fn(x, sigma=sigmas[num_steps-1-i])
            x_pred = x * alpha - x_denoised * beta
            x_eps = x * beta + x_denoised * alpha

            #if not is_last:
            alpha, beta = self.get_alpha_beta(sigmas[num_steps-2-i].item())
            x = x_pred * alpha + x_eps * beta
            print("std of x aaaaaaaaaaaoooooooooooooooo", x.std().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            print("sigma", sigmas[num_steps-1-i].item())
        print(sigmas)

        return x


class KarrasSampler(Sampler):
    """https://arxiv.org/abs/2206.00364 algorithm 1"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
        bridge: bool = True
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn
        self.bridge = bridge

    def step(
        self, x: Tensor, fn: Callable, sigma: float, sigma_next: float, gamma: float
    ) -> Tensor:
        """Algorithm 2 (step)"""
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        # Evaluate ∂x/∂sigma at sigma_hat
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d
        # Second order correction
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_prime)
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
    
        if self.bridge:
            x = noise
        elif not self.bridge:
            x = sigmas[0] * noise 
        else:
            print("Select if you want to performe schrodinger bridge or not") 

        self.s_churn = 0
        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, sqrt(2) - 1),
            0.0,
        )
        # Denoise to sample
        for i in range(num_steps - 1):
            #file_path = f"/home/demancum/Samples/x_test.wav"
            #torchaudio.save(file_path, x[0].cpu(), 16000)
            x = self.step(
                x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1], gamma=gammas[i]  # type: ignore # noqa
            )
            #print("std of x", x.std().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            #print("sigmas  ", sigmas[i].item())


        return x

class KarrasSamplerReverse(Sampler):
 

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(
        self,
        s_tmin: float = 0,
        s_tmax: float = float("inf"),
        s_churn: float = 0.0,
        s_noise: float = 1.0,
    ):
        super().__init__()
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.s_churn = s_churn

    def step(
        self, x: Tensor, fn: Callable, sigma: float, sigma_next: float, gamma: float
    ) -> Tensor:
        """Algorithm 2 (step)"""  #gamma uguale a 0 
        # Select temporarily increased noise level
        sigma_hat = sigma + gamma * sigma  
        # Add noise to move from sigma to sigma_hat
        epsilon = self.s_noise * torch.randn_like(x)
        x_hat = x + sqrt(sigma_hat ** 2 - sigma ** 2) * epsilon
        # Evaluate ∂x/∂sigma at sigma_hat
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        # Take euler step from sigma_hat to sigma_next
        x_next = x_hat + (sigma_next - sigma_hat) * d
        # Second order correction
        if sigma_next != 0:
            model_out_next = fn(x_next, sigma=sigma_next)
            d_prime = (x_next - model_out_next) / sigma_next
            x_next = x_hat + 0.5 * (sigma_next - sigma_hat) * (d + d_prime)
        return x_next

    def forward(
        self, x: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        self.s_churn = 0
        # Compute gammas
        gammas = torch.where(
            (sigmas >= self.s_tmin) & (sigmas <= self.s_tmax),
            min(self.s_churn / num_steps, sqrt(2) - 1),
            0.0,
        )
        # Add noise up to gaussian (from num_steps to 0)
        for i in range(num_steps - 1):
            #file_path = f"/home/demancum/Slides/noise_flute_{i}.wav"
            #torchaudio.save(file_path, x[0].cpu(), 16000)
            x = self.step(
                x, fn=fn, sigma=sigmas[num_steps-1-i], sigma_next=sigmas[num_steps-2-i], gamma=gammas[i]  # type: ignore # noqa
            )
            # print(i)
            # print("std of x", x.std().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            # print("sigmas  ", sigmas[num_steps-1-i].item())
            # print(" ")
        return x


class AEulerSampler(Sampler):

    diffusion_types = [KDiffusion, VKDiffusion]

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float]:
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        return sigma_up, sigma_down

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Euler method
        x_next = x + d * (sigma_down - sigma)
        # Add randomness
        x_next = x_next + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x

class ADPM2SamplerReverse(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, sigmas, rho: float = 1.0):
        super().__init__()
        self.rho = rho
        self.sigmas = sigmas
        self.sigma_up_list = []
        self.sigma_down_list = []
        self.sigma_mid_list = []
        self.num_steps = len(sigmas) - 1

        for i in range(self.num_steps):
 
            r = 1
            sigma_up = sqrt(sigmas[i+1] ** 2 * (sigmas[i] ** 2 - sigmas[i+1] ** 2) / sigmas[i] ** 2)
            sigma_down = sqrt(sigmas[i+1] ** 2 - sigma_up ** 2)
            sigma_mid = ((sigmas[i] ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
            
            self.sigma_up_list.append(sigma_up)
            self.sigma_down_list.append(sigma_down)
            self.sigma_mid_list.append(sigma_mid)


    def step(self, x: Tensor, i, fn: Callable) -> Tensor:
        # Sigma steps
        z = self.num_steps - 1 - i
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=self.sigmas[z])) / self.sigmas[z]
        # Denoise to midpoint
        x_mid = x + d * ( - self.sigma_mid_list[z] + self.sigmas[z])
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=self.sigma_mid_list[z])) / self.sigma_mid_list[z]
        # Denoise to next
        x = x + d_mid * ( - self.sigma_down_list[z] + self.sigmas[z])
        # Add randomness
        #x_next = x + torch.randn_like(x) * sigma_up
        return x

    def forward(
        self, x: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        
        # Add noise
        for i in range(num_steps - 1):
            x = self.step(x, i, fn=fn)  # type: ignore # noqa
            print("std of x", x.std().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            #print("sigma", sigmas[i].item())
        return x

    
class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types = [KDiffusion, VKDiffusion]

    def __init__(self, bridge: bool = False, rho: float = 1.0):
        super().__init__()
        self.rho = rho
        self.bridge = bridge

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        #x_next = x + torch.randn_like(x) * sigma_up
        return x

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        
        if self.bridge:
            x = noise
        elif not self.bridge:
            x = sigmas[0] * noise 
        else:
            print("Select if you want to performe schrodinger bridge or not")

        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
            print("std of x", x.std().item()) #check std of the sample at each step should be similar to sigma (only for big sigmas)
            print("sigma", sigmas[i].item())
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        for i in range(num_steps - 1):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask


""" Main Classes """


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = None,
        clamp: bool = False, 
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    @torch.no_grad()
    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = default(num_steps, self.num_steps)  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        
        sigmas = self.sigma_schedule(num_steps, device)


        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x 
        return x


class DiffusionInpainter(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        num_steps: int,
        num_resamples: int,
        sampler: Sampler,
        sigma_schedule: Schedule,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.num_steps = num_steps
        self.num_resamples = num_resamples
        self.inpaint_fn = sampler.inpaint
        self.sigma_schedule = sigma_schedule

    @torch.no_grad()
    def forward(self, inpaint: Tensor, inpaint_mask: Tensor) -> Tensor:
        x = self.inpaint_fn(
            source=inpaint,
            mask=inpaint_mask,
            fn=self.denoise_fn,
            sigmas=self.sigma_schedule(self.num_steps, inpaint.device),
            num_steps=self.num_steps,
            num_resamples=self.num_resamples,
        )
        return x


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[2], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[:, :, start:] = torch.zeros((length - start,), device=device)
    return mask


class SpanBySpanComposer(nn.Module):
    def __init__(
        self,
        inpainter: DiffusionInpainter,
        *,
        num_spans: int,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.num_spans = num_spans

    def forward(self, start: Tensor, keep_start: bool = False) -> Tensor:
        half_length = start.shape[2] // 2

        spans = list(start.chunk(chunks=2, dim=-1)) if keep_start else []
        # Inpaint second half from first half
        inpaint = torch.zeros_like(start)
        inpaint[:, :, :half_length] = start[:, :, half_length:]
        inpaint_mask = sequential_mask(like=start, start=half_length)

        for i in range(self.num_spans):
            # Inpaint second half
            span = self.inpainter(inpaint=inpaint, inpaint_mask=inpaint_mask)
            # Replace first half with generated second half
            second_half = span[:, :, half_length:]
            inpaint[:, :, :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=2)


class XDiffusion(nn.Module):
    def __init__(self, type: str, net: nn.Module, **kwargs):
        super().__init__()

        diffusion_classes = [VDiffusion, KDiffusion, VKDiffusion]
        aliases = [t.alias for t in diffusion_classes]  # type: ignore
        message = f"type='{type}' must be one of {*aliases,}"
        assert type in aliases, message
        self.net = net

        for XDiff in diffusion_classes:
            if XDiff.alias == type:  # type: ignore
                self.diffusion = XDiff(net=net, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(diff_self=self, *args, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        clamp: bool = False,
        **kwargs,
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
            clamp=clamp,
        )
        return diffusion_sampler(noise, **kwargs)

    def log_tensorboard_audio(self, writer, id, samples, sampling_rate, step):
        num_items = samples.shape[0]
        # samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        for idx in range(num_items):
            if idx < 4: # limit the examples to 4
                writer.add_audio(f"{id}/sample_{idx}", samples[idx], step, sample_rate=sampling_rate)
            else:
                break
    def log_tensorboard_spectrogram(self, writer, id, samples, sampling_rate, step, sigmas=None):
        num_items = samples.shape[0]
        samples = samples.detach().cpu()
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=80,
            center=True,
            norm="slaney",
        )

        for idx in range(num_items):
            if idx < 4: # limit the examples to 4
                spectrogram = transform(samples[idx][0])
                fig, ax = plt.subplots()
                img = librosa.display.specshow(librosa.power_to_db(spectrogram), sr=sampling_rate, hop_length=512, ax=ax, x_axis='time', y_axis='mel')
                # check if the variable sigmas exists
                if isinstance(sigmas, torch.Tensor):
                    ax.set(title=f'sigma={sigmas[idx]}')
                else:
                    ax.set(title='Mel spectrogram')
                writer.add_figure(f"{id}/mel_spectrogram_{idx}", fig, step)
            else:
                break
