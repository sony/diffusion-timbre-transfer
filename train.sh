python train.py \
exp=flute_latent \
trainer.gpus=1 \
model.lr=1e-4 \
trainer.precision=32 \
trainer.max_epochs=500 \
datamodule.batch_size=32 \
datamodule.num_workers=16 \
+dataset_path=/mnt/beegfs/group/mfm/data/cocochorales_mini/main_dataset/train \
exp_tag=pitch_flute_s100