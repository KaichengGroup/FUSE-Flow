import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

from data_modules.npz_dataset import NPZDataset
from utils.utils import load_config, create_subset, save_train_results, CONFIG_PATH, determine_version_name, \
    get_latest_checkpoint_path, CONFIG_FILENAME, save_outputs, initialize_model

if __name__ == '__main__':
    pl.seed_everything(42)

    # "highest" (default), float32 matrix multiplications use the float32 datatype for internal computations.
    # "high", float32 matrix multiplications use the TensorFloat32 or bfloat16_3x
    # "medium", float32 matrix multiplications use the bfloat16 datatype
    torch.set_float32_matmul_precision('highest')

    config = load_config(CONFIG_PATH)

    # instantiate trainer
    trainer = Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None
    )

    # instantiate model
    dataset_dir = os.path.join(config['data']['log_dir'], config['data']['dataset'])
    version_name = determine_version_name(
        dataset_dir,
        config['testing']['run_name'],
        False
    )
    version_dir = os.path.join(dataset_dir, version_name)
    checkpoint_path = get_latest_checkpoint_path(version_dir)
    save_train_results(version_dir)
    model_config = load_config(os.path.join(version_dir, CONFIG_FILENAME))

    # initiate testing dataloader
    test_dataset = NPZDataset(
        root=os.path.join(config['data']['data_dir'], config['data']['dataset']),
        filename=config['testing']['filename'],
        transform=Compose([ToTensor()])
    )
    test_subset = create_subset(
        dataset=test_dataset,
        sample_size=config['testing']['sample_size']
    )
    test_dataloader = DataLoader(
        dataset=test_subset,
        batch_size=config['testing']['batch_size'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0
    )

    temperatures = config['testing']['temperature']
    temperatures = [temperatures] if not isinstance(temperatures, list) else temperatures

    # instantiate model
    for temperature in temperatures:
        model = initialize_model(
            config=config,
            model_config=model_config,
            temperature=temperature,
            version_name=version_name,
            checkpoint_path=checkpoint_path,
            input_shape=test_dataset[0][0].shape,
            output_shape=test_dataset[0][1].shape
        )

        outputs = trainer.predict(model, dataloaders=test_dataloader)
        save_outputs(
            dataset_dir=version_dir,
            outputs=outputs,
            batch_size=config['testing']['batch_size'],
            npy_name=f"{config['testing']['filename']}_{temperature}.npy"
        )
