import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

from FUSE_Flow.fuse_flow import FUSEFlow
from data_modules.npz_dataset import NPZDataset
from utils.utils import load_config, create_subset, determine_version_name, CONFIG_PATH, copy_config, copy_code

if __name__ == '__main__':
    pl.seed_everything(42)

    # "highest" (default), float32 matrix multiplications use the float32 datatype for internal computations.
    # "high", float32 matrix multiplications use the TensorFloat32 or bfloat16_3x
    # "medium", float32 matrix multiplications use the bfloat16 datatype
    torch.set_float32_matmul_precision('highest')

    config = load_config(CONFIG_PATH)

    # initiate training dataloader
    train_dataset = NPZDataset(
        root=os.path.join(config['data']['data_dir'], config['data']['dataset']),
        filename=config['training']['filename'],
        transform=Compose([ToTensor()])
    )
    train_subset = create_subset(
        dataset=train_dataset,
        sample_size=config['training']['sample_size']
    )
    train_loader = DataLoader(
            dataset=train_subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
    )

    # instantiate trainer
    dataset_dir = os.path.join(config['data']['log_dir'], config['data']['dataset'])
    version_name = determine_version_name(
        dataset_dir,
        config['training']['run_name'],
        True
    )

    # instantiate model
    model = FUSEFlow(
        input_shape=train_dataset[0][0].shape,
        output_shape=train_dataset[0][1].shape,
        ablation=config['ablation'],
        hyper=config['hyper-parameters'],
        temperature=config['hyper-parameters']['prior_variance'],
        augmentations=config['augmentations']
    )

    # save configurations
    copy_config(config['data']['log_dir'], config['data']['dataset'], version_name)

    # save code snapshot
    copy_code(config['data']['log_dir'], config['data']['dataset'], version_name)

    # train model
    try:
        trainer = Trainer(
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=config['training']['epochs'],
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                ModelCheckpoint(dirpath=None, every_n_epochs=1),
                # ModelSummary(max_depth=4)
            ],
            logger=CSVLogger(
                save_dir=config['data']['log_dir'],
                name=config['data']['dataset'],
                version=version_name,
                flush_logs_every_n_steps=config['training']['flush']
            ),
            log_every_n_steps=config['training']['log_interval'],
            enable_model_summary=True
        )
        trainer.fit(model, train_loader)
    except AssertionError:
        trainer = Trainer(
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=config['training']['epochs'],
            gradient_clip_val=1.0,
            enable_checkpointing=True,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                ModelCheckpoint(dirpath=None, every_n_epochs=1),
                # ModelSummary(max_depth=4)
            ],
            logger=CSVLogger(
                save_dir=config['data']['log_dir'],
                name=config['data']['dataset'],
                version=version_name,
                flush_logs_every_n_steps=config['training']['flush']
            ),
            log_every_n_steps=config['training']['log_interval'],
            enable_model_summary=False
        )
        trainer.fit(model, train_loader)
