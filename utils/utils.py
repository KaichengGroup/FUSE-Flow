import importlib.util
import math
import os
import shutil

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import Subset

from FUSE_Flow.fuse_flow import FUSEFlow

CONFIG_FILENAME = 'configurations.yaml'
CONFIG_PATH = os.path.join('configurations.yaml')

SRC_CODE_DIR = 'FUSE_Flow'


def determine_version_name(dataset_dir, run_name, to_increment):
    if run_name is not None:
        version_name = str(run_name)
    else:
        os.makedirs(dataset_dir, exist_ok=True)
        past_runs = os.listdir(dataset_dir)
        current_versions = [int(folder_name) for folder_name in past_runs if str(folder_name).isnumeric()]
        if len(current_versions) > 0:
            version_name = str(max(current_versions) + int(to_increment))
        else:
            version_name = '0'
    return version_name


def get_latest_checkpoint_path(version_dir):
    checkpoint_dir = os.path.join(str(version_dir), 'checkpoints')
    latest_checkpoint = sorted(os.listdir(checkpoint_dir),
                               key=lambda x: int(x.split('.')[0].split('-')[1].split('=')[1]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    return checkpoint_path


def copy_config(log_dir, dataset_name, version_name):
    version_dir = os.path.join(log_dir, dataset_name, version_name)
    os.mkdir(version_dir)
    shutil.copyfile(CONFIG_PATH, os.path.join(version_dir, CONFIG_FILENAME))


def copy_code(log_dir, dataset_name, version_name):
    version_dir = os.path.join(log_dir, dataset_name, version_name)
    shutil.copytree(SRC_CODE_DIR, os.path.join(version_dir, SRC_CODE_DIR))


def save_outputs(dataset_dir, outputs, batch_size, npy_name):
    n_posterior = len(outputs[0])//batch_size
    output_shape = np.array(outputs[0].shape[1:]).squeeze()
    n_samples = (len(outputs) - 1) * batch_size + len(outputs[-1]) // n_posterior
    sr_array = np.empty((n_samples, n_posterior, *output_shape), dtype=np.uint8)

    for i, sr_batch in enumerate(outputs):
        curr_batch_size = len(sr_batch) // n_posterior
        for j in range(n_posterior):
            posterior_sample = sr_batch[(j * curr_batch_size):((j + 1) * curr_batch_size)].numpy().astype(np.uint8)
            sr_array[(i * batch_size):(i * batch_size + curr_batch_size), j] = posterior_sample

    np.save(os.path.join(dataset_dir, npy_name), sr_array)


def load_config(filename):
    with open(filename, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def create_subset(dataset, sample_size):
    """Returns a subset of a PyTorch dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or Union
    sample_size : float or int
        Represents number or proportion of original images (not patches) to sample.
        If float, assumes between 0 and 1, which represents proportion.

    Returns
    -------
    subset : torch.utils.data.Dataset
    """
    if sample_size is None:
        subset = dataset
    else:
        if isinstance(sample_size, int):
            subset_size = min(sample_size, len(dataset))
        elif 0 < sample_size < 1:
            subset_size = int(sample_size*len(dataset))
        else:
            print(f'Indeterminate subset size. Using full dataset. '
                  f'Got {str(sample_size)}. '
                  f'Only accept integer [0,{str(len(dataset))}] or '
                  f'float [0.0, 1.0].')
            subset_size = len(dataset)
        if isinstance(dataset, list):
            subset = dataset[:subset_size]
        else:
            subset = Subset(dataset, list(range(subset_size)))
    return subset


def save_train_results(root):
    """

    Parameters
    ----------
    root
    """
    os.makedirs(root, exist_ok=True)
    _save_loss_curve(root)


def initialize_model(config, model_config, temperature, version_name,
                     checkpoint_path, input_shape, output_shape):
    """Initializes trained model for testing.
    Uses code snapshot that should match weight structure if available, else uses master code.

    Parameters
    ----------
    config : dict
    model_config : dict
    temperature : float
    version_name : str
    checkpoint_path : str
    input_shape : tuple
    output_shape : tuple

    Returns
    -------
    model : pl.LightningModule
    """
    try:
        # specify the module that needs to be imported relative to the path of the module
        spec = importlib.util.spec_from_file_location(
            'FUSE_Flow',
            os.path.join(
                config['data']['log_dir'],
                config['data']['dataset'],
                version_name,
                'FUSE_Flow',
                'fuse_flow.py'
            )
        )
        # creates a new module based on spec
        snapshot_module = importlib.util.module_from_spec(spec)
        # executes the module in its own namespace when a module is imported or reloaded.
        spec.loader.exec_module(snapshot_module)

        model = snapshot_module.FUSEFlow.load_from_checkpoint(
            checkpoint_path,
            input_shape=input_shape,
            output_shape=output_shape,
            ablation=model_config['ablation'],
            hyper=model_config['hyper-parameters'],
            temperature=temperature,
            augmentations=model_config['augmentations'],
            sample_size=config['testing']['posterior_sample_size']
        )
        print('Model loaded from snapshot.')
    except FileNotFoundError:
        try:
            model = FUSEFlow.load_from_checkpoint(
                checkpoint_path,
                input_shape=input_shape,
                output_shape=output_shape,
                ablation=model_config['ablation'],
                hyper=model_config['hyper-parameters'],
                temperature=temperature,
                augmentations=model_config['augmentations'],
                sample_size=config['testing']['posterior_sample_size']
            )
            print('Code snapshot not found. Model loaded from master.')
        except KeyError:
            print('Code snapshot not found. Weights mismatch. Aborting training...')
            exit()
    except TypeError:
        print('Code snapshot found but weights mismatch. Aborting training...')
        exit()
    return model


def _save_loss_curve(root):
    """Save loss curve. Also saves a magnified version of last 10% of steps.

    Parameters
    ----------
    root : str
    """
    try:
        metrics = pd.read_csv(os.path.join(root, 'metrics.csv'))
        df_loss = metrics[~metrics.loss.isnull()]
        df_lr = metrics[metrics.loss.isnull()]
        loss_path = os.path.join(root, 'loss_curve.png')
        plt.plot(df_loss['step'], df_loss['loss'])
        plt.annotate('lr:{:.1e}'.format(df_lr['lr-Adam'].iloc[0]),
                     (df_loss['step'].iloc[0], df_loss['loss'].iloc[0]))
        plt.annotate('lr:{:.1e}'.format(df_lr['lr-Adam'].iloc[-1]),
                     (df_loss['step'].iloc[-2], df_loss['loss'].iloc[-2]))
        plt.ylabel('loss', fontsize=16)
        plt.savefig(loss_path)
        plt.close()
        loss_path = os.path.join(root, 'log_loss_curve.png')
        plt.plot(df_loss['step'], np.log(df_loss['loss']))
        plt.annotate('lr:{:.1e}'.format(df_lr['lr-Adam'].iloc[0]),
                     (df_loss['step'].iloc[0], np.log(df_loss['loss']).iloc[0]))
        plt.annotate('lr:{:.1e}'.format(df_lr['lr-Adam'].iloc[-1]),
                     (df_loss['step'].iloc[-2], np.log(df_loss['loss']).iloc[-2]))
        plt.ylabel('log-loss', fontsize=16)
        plt.savefig(loss_path)
        plt.close()
        loss_path = os.path.join(root, 'loss_curve_magnified.png')
        plt.plot(df_loss['step'].iloc[-math.floor(len(df_loss) * 0.1):],
                 df_loss['loss'].iloc[-math.floor(len(df_loss) * 0.1):])
        plt.ylabel('loss', fontsize=16)
        plt.savefig(loss_path)
        plt.close()
    except FileNotFoundError:
        pass
