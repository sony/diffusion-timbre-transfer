# Latent Diffusion Bridges for Unsupervised Musical Audio Timbre Transfer

This codebase is for the following [paper](https://arxiv.org/abs/2409.06096). Pretrained model weights and computed Encodec mean and std tensors are available [here](https://zenodo.org/records/13849169).
## Diffusion Bridges
An example of running inference for diffusion brindges can be found in `inference.ipynb`. 
## Setup

(Optional) Create virtual environment and activate it

```bash
conda create --name diff python=3.10

conda activate diff
```
Install requirements

```bash
pip install -r requirements.txt
```

Git clone the repository basic-pitch-torch to the audio_diffusion_pytorch directory of the project

```bash
git clone https://github.com/gudgud96 basic-pitch-torch
```


## Training

In the `exp` folder, you will find several subfolders that contain configuration files used across all experiments. These files define general settings like training, model parameters, and logging, which are independent of any specific experiment. Below is a brief overview of these folders:

- **callbacks**: Contains configurations for various callbacks, such as model checkpoints, model summaries, and loggers.
- **datamodule**: Defines data-related configurations, including the validation split and which dataset class to use. By default, it uses `audio_data_pytorch.WaVDataset` to work with `.wav` files, but you can create and specify a custom dataset here. Data transforms are also specified in this folder.
- **loggers**: By default, TensorBoard is used for logging, but you can add or customize the logger configuration here.
- **model**: Contains the model configuration files, defining the model architecture and parameters.
- **trainer**: Defines training configurations, such as GPUs to use, precision settings, number of epochs, etc.

### Creating your own configuration
To create a new experiment, you need to add a new `.yaml` file in the `exp` folder where you specify the experimental settings. These settings can include parameters like the instruments used, sigma_min, sigma_max, etc. Alternatively, you can use existing configurations, such as `flute_latent.yaml`, to train specific models, or modify them according to your needs.

Note, before using an existing .yaml config, change mean_path and std_path to the dir where you have the right mean and std tensors.

### Dataset Path
When specifying the `dataset_path`, ensure it points to a directory that contains `.wav` files. The files should have the instrument name included in the filename (e.g., `flute1.wav`). The system will recursively search through all subdirectories for such `.wav` files based on the instrument name.

### Running train.py


```bash
python train.py \
  exp=name_of_your_yaml_file_in_exp_folder \
  trainer.gpus=1 \
  model.lr=1e-4 \
  trainer.precision=32 \
  trainer.max_epochs=500 \
  datamodule.batch_size=32 \
  datamodule.num_workers=16 \
  +dataset_path=path/to/your/data \
  exp_tag=name_of_exp
```

### Resume run from a checkpoint
If you want to resume training from a previous checkpoint, uncomment the ckpt: line in the corresponding config.yaml file and provide the path to the .ckpt file. By default, checkpoints generated during training will be saved in the logs/name_of_exp directory.

## Saving a checkpoint
By default, a log directory is created in the root folder, and for each experiment, a subfolder with the experiment's name is generated. During training, checkpoints are automatically saved in this subfolder.
