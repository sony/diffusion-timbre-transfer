from typing import Any, Callable, List, Optional
import os
import librosa
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import AudioDiffusionModel, AudioDiffusionConditional, Sampler, Schedule
from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torch import Tensor, nn
from torch.utils.data import DataLoader
#from encodec import EncodecModel
from transformers import EncodecModel
#DEFAULT_ENCODEC_MODEL = EncodecModel.from_pretrained("facebook/encodec_24khz")
#DEFAULT_ENCODEC_MODEL = DEFAULT_ENCODEC_MODEL.to('cuda')
import matplotlib.pyplot as plt
import pandas as pd

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        mean_path: str,
        std_path: str,
        lr: float = 1e-4,
        lr_beta1: float = 0.95,
        lr_beta2: float = 0.999,
        lr_eps: float = 1e-6,
        lr_weight_decay: float = 1e-3,
        total_steps: float = None,
        ema_beta: float = 0.995,
        ema_power: float = 0.7,
        latent: bool = True,
        cond: bool = False,
        encodec_model: str = 'facebook/encodec_24khz',
        sampling_rate: int = 24000,
        #embedder: nn.Module
        #encodec_model: EncodecModel = DEFAULT_ENCODEC_MODEL
        
    ):
        
        super().__init__()
        #encodec_model = EncodecModel.encodec_model_24khz()
        #encodec_model.set_target_bandwidth(6.0)
        #self.encoder = encodec_model
    
        #encodec_model = encodec_model.to('cuda')
        #encodec_model.requires_grad_(False)
        #self.encoder = encodec_model.get_encoder()
        #self.decoder = encodec_model.get_decoder()

        self.latent = latent
        self.cond = cond

        if self.cond == "encodec":
            self.encodec_model = EncodecModel.from_pretrained(encodec_model) #.to('cuda')
            self.encodec_model.requires_grad_(False)
        elif self.cond == "label":
            # embedding layer for instrument label
            self.instrument_to_idx = {
                'violin': 0,
                'flute': 1,
                'bassoon': 2,
                'clarinet': 3,
                'cello': 4,
                'trumpet': 5,
                'oboe': 6,
                'horn': 7,
                'viola': 8,
                'tuba': 9,
                'trombone': 10,
                'saxophone': 11,
                'double bass': 12}
            self.embedding = nn.Embedding(13, 128)        

        if self.cond == False and self.latent == True:
            self.encodec_model = EncodecModel.from_pretrained(encodec_model) #.to('cuda')
            self.encodec_model.requires_grad_(False)
            mean = torch.load(mean_path)
            std = torch.load(std_path)
            self.mean = rearrange(mean, "i -> i 1")
            self.std = rearrange(std, "i -> i 1")

        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.total_steps = total_steps
        # Diffusion Model
        self.model = model
        self.sampling_rate = sampling_rate
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)
        # Text Encoder
        #self.embedder = embedder
        dev1 = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize variables to store losses and sigmas
        self.iteration_losses_before = torch.Tensor().to(dev1)
        self.iteration_losses = torch.Tensor().to(dev1)
        self.iteration_sigmas = torch.Tensor().to(dev1)
        self.iteration_counter = 0

        # flag for logging
        self.log_next = False

    @torch.no_grad()
    def encode_latent(self, x: Tensor) -> Tensor:
        encoder = self.encodec_model.get_encoder()
        z = encoder(x) 
        if self.cond == False and self.latent == True:
            z = (z-self.mean.to(z.device))/self.std.to(z.device)
        return z
    
    @torch.no_grad()
    def decode_latent(self, z: Tensor) -> Tensor:
        decoder = self.encodec_model.get_decoder()
        if self.cond == False and self.latent == True:
            z = self.mean.to(z.device) + z * self.std.to(z.device)
        x = decoder(z)
        
        return x




    def record_iteration_data(self, losses_before, losses, sigmas):
        """Record losses and sigmas for each iteration."""
        self.iteration_losses_before = torch.cat((self.iteration_losses_before, losses_before))
        self.iteration_losses = torch.cat((self.iteration_losses, losses))
        self.iteration_sigmas = torch.cat((self.iteration_sigmas, sigmas))

    def save_iteration_data(self):
        """Save the recorded losses and sigmas, updating the original CSV file."""

        data = torch.stack((self.iteration_losses_before, self.iteration_losses, self.iteration_sigmas))
        data_numpy = data.detach().cpu().numpy()
        data_numpy_transposed = data_numpy.T
        new_df = pd.DataFrame(data_numpy_transposed, columns=['Losses_before', 'Losses', 'Sigmas'])

        file_path = '/speech/dbwork/mul/spielwiese4/students/demancum/latent_flute_channel_norm_KDIST_mean=0_std=1.0_patch.csv'

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df

        updated_df.to_csv(file_path, index=False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        if self.total_steps:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_steps)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}] 
        else:
            return optimizer

    def on_validation_epoch_start(self):
        self.log_next = True        

    def training_step(self, batch, batch_idx):
        # batch = (waveforms, inst_label)

        # if global step is a multiple of val_check_interval
        # This is to reduce the logging frequency for training
        if self.global_step % self.trainer.val_check_interval == 0:
            self.log_next = True
        else:
            self.log_next = False

        if self.latent == True:
            with torch.no_grad():
                waveforms = self.encode_latent(batch[0])
                
                embedding = None 
        elif self.latent == False:
            if self.cond == 'encodec':
                embedding = self.encode_latent(batch[0])
                embedding = rearrange(embedding, "i j k -> i k j")
            elif self.cond == 'label':
                labels = [self.instrument_to_idx[inst_name] for inst_name in batch[1]]
                labels = torch.tensor(labels).to(batch[0].device)
                embedding = self.embedding(labels)
                # make it to the same shape as encodec
                embedding = embedding.unsqueeze(1).expand(-1, 1280, -1)
            elif self.cond == False:
                embedding = None
            
            else:
                print("Please choose cond between True of False")
            
            waveforms = batch[0]
                       
        else:
            print("Please choose latent between True of False")

        loss, losses, sigmas = self.model(
            waveforms,
            embedding=embedding,
            pl_self = self)
        self.log("train_loss", loss)
        # Update EMA model and log decay
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay())

        #self.iteration_counter += 1
        #losses_before = torch.zeros_like(losses).to('cuda')

        #if self.iteration_counter >= 500:
        #    self.record_iteration_data(losses_before, losses, sigmas)

        #if self.iteration_counter % 1000 == 0:
        #    self.save_iteration_data()
        if self.log_next:
            self.log_next = False
        
        return loss

        

    def validation_step(self, batch, batch_idx):
        if self.latent == True:
            with torch.no_grad():
                waveforms = self.encode_latent(batch[0])
                
                embedding = None 
                
        elif self.latent == False:
            if self.cond == 'encodec':
                embedding = self.encode_latent(batch[0])
                embedding = rearrange(embedding, "i j k -> i k j")
            elif self.cond == 'label':
                labels = [self.instrument_to_idx[inst_name] for inst_name in batch[1]]
                labels = torch.tensor(labels).to(batch[0].device)
                embedding = self.embedding(labels)
                # make it to the same shape as encodec
                embedding = embedding.unsqueeze(1).expand(-1, 1280, -1)
            elif self.cond == False:
                embedding = None
            
            else:
                print("Please choose cond between True of False")
            
            waveforms = batch[0]
        
        else:
            print("Please choose latent between True of False")

    
        loss, losses, sigmas = self.model_ema(
            waveforms,
            embedding=embedding,
            pl_self = self)

        self.log("valid_loss", loss)
        if self.log_next:
            self.log_next = False        
        return loss


""" Datamodule """

#def worker_init_fn(worker_id):
#    global encodec_model
#    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to('cuda')


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            #worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            #worker_init_fn=worker_init_fn
        )


""" Callbacks """


# def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
#     """Safely get Weights&Biases logger from Trainer."""

#     if isinstance(trainer.logger, WandbLogger):
#         return trainer.logger

#     if isinstance(trainer.logger, LoggerCollection):
#         for logger in trainer.logger:
#             if isinstance(logger, WandbLogger):
#                 return logger

#     print("WandbLogger not found.")
#     return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
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

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )

class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        diffusion_schedule: Schedule,
        diffusion_sampler: Sampler,
        use_ema_model: bool,
        latent: bool,
        cond: bool,
        mean_path: str,
        std_path: str,
        #encodec_model: EncodecModel = DEFAULT_ENCODEC_MODEL
    ) -> None:

        #encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        #self.encodec_model = encodec_model.to('cuda')
        #self.encodec_model.requires_grad_(False)
        #self.encoder = self.encodec_model.get_encoder()
        #self.decoder = encodec_model.get_decoder()

        #self.bandwidth = 24.0
        #self.scales = [None]

        self.latent = latent
        self.cond = cond

        if self.cond == False and self.latent == True:
            mean = torch.load(mean_path)
            std = torch.load(std_path)

            self.mean = rearrange(mean, "i -> i 1")
            self.std = rearrange(std, "i -> i 1")            

        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_sampler = diffusion_sampler
        self.use_ema_model = use_ema_model
        if self.latent == True: #TODO check for audio and latent
            self.clamp = False
        elif self.latent == False:
            self.clamp = True
        
        else:
            print("Please choose latent between True of False")

        self.log_next = False

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        # Check if trainer.logger is a single logger or a collection
        if isinstance(trainer.logger, TensorBoardLogger):
            tensorboard_logger = trainer.logger
        # elif isinstance(trainer.logger, LoggerCollection):
        #     tensorboard_logger = [logger for logger in trainer.logger if isinstance(logger, TensorBoardLogger)][0]
        else:
            raise ValueError("TensorBoardLogger not found in the trainer logger collection.")

        writer = tensorboard_logger.experiment

        diffusion_model = pl_module.model
        if self.use_ema_model:
            diffusion_model = pl_module.model_ema.ema_model

        

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )
        
        if self.latent == True: #TODO check for audio and latent
            z = pl_module.encode_latent(noise)
            noise = torch.randn_like(z)
            embedding = None
             
        elif self.latent == False:
            if self.cond == True:
                embedding = pl_module.encode_latent(batch[:self.num_items,:,:])
                embedding = rearrange(embedding, "i j k -> i k j")
            
            elif self.cond == False:
                embedding = None
            
            else:
                print("Please choose cond between True of False")

            noise = noise
        
        else:
            print("Please choose latent between True of False")

        #noise_ori = torch.randn( 
        #    (self.num_items, 1, 409600), device=pl_module.device
        #)

        #noise = self.encoder(noise_ori)
        #noise = torch.load('/speech/dbwork/mul/spielwiese4/students/demancum/noise.pt') #TODO TOGLIERE
        #noise.to("cuda")

        for steps in self.sampling_steps:
            samples = diffusion_model.sample(
                noise=noise,
                embedding=embedding,
                sampler=self.diffusion_sampler,
                sigma_schedule=self.diffusion_schedule,
                num_steps=steps,
                clamp=self.clamp
            )
            
            #samples = samples*5.23 + (-0.56) #TODO REMOVE

            if self.latent == True: #TODO check for audio and latent

                #codes = self.encodec_model.quantizer.encode(samples, self.bandwidth)
                #codes = codes.transpose(0, 1)
                #codes = codes.unsqueeze(0)
                #waveform = self.encodec_model.decode(codes, self.scales)
                #samples = waveform.audio_values.squeeze(0)

                #samples = self.mean.to(samples.device) + samples * self.std.to(samples.device) 
                waveform = pl_module.decode_latent(samples)
                samples = waveform.squeeze(0)
            
            elif self.latent == False:
                samples = samples
            
            else:
                print("Please choose latent between True of False")

            pl_module.model.diffusion.log_tensorboard_audio(
                writer=writer,
                id=f"sampling_steps-{steps}",
                samples=samples,
                sampling_rate=self.sampling_rate,
                step=trainer.global_step
            )
            pl_module.model.diffusion.log_tensorboard_spectrogram(
                writer=writer,
                id=f"sampling_steps-{steps}",
                samples=samples,
                sampling_rate=self.sampling_rate,
                step=trainer.global_step
            )
        if is_train:
            pl_module.train()
