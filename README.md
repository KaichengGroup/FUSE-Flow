# FUSE-Flow

## Getting Started

### _configurations.yaml_

Hyperparameters, architecture modifications, and configurations for training and testing are defined 
in a _configurations.yaml_ text file. The file is _.gitignore_'ed for commit cleanliness, so you'll 
have to create your own. Below are the defaults.
```yaml
ablation:
  no_pretrain: false  # do not load weights of pre-trained u-net from models/pretrain_unet/weights.pth
  no_freeze: false  # do not freeze u-net during training of normalizing flow
  no_flow: false  # no normalizing flow, only u-net
  no_autoencoder: false  # no u-net, only normalizing flow
  no_augmentation: false  # no augmentations as defined in data_modules/augmentation.py
  no_transition: false  # no transitional steps as defined in SRFlow 
  no_actnorm: false  # no actnorm in flow step
  no_1x1_conv: false  # no invertible 1x1 convolution in flow step
  no_injection: false  # no affine injection in flow step
  no_coupling: false  # no affine coupling in flow step
  logistic_coupling: false  # use logistic coupling as defined in Flow++. this isn't working yet
  no_skip: false  # no skip connections in u-net
  autoencoder_loss: bce  # loss function for u-net if no normalizing flow. accepts: bce, l1, l2
  dequantization: var  # either variation dequantization, basic floor dequantization, or none. accepts: var, basic, none
  attention_type: se  # type of self-attention mechanism. accepts: se, cbam, none
augmentations:
  hor_flip: true  # horizontal flip
  ver_flip: false  # vertical flip
  col_jig: false  # color jiggle
hyper-parameters:
  factor: 2  # factor at which dimensions are scaled. currently only works with 2
  flow:
    n_scale_add: 0  # additional reduction in dimensions to in mapping to prior beyond what is minimally required. max value for CelebA is 1, max value for CIFAR10 is 6.
    n_step: 4  # number of flow steps in each scale block
  autoencoder:
    c_u: 128  # number of feature channels passed from u-net to normalizing flow
    n_conv: 32  # number of gated_convolutions in gated_resnet in u-net
    attn_red_ratio: 0.5  # 0-1 slider to adjust influence of attention mechanism
  dequantization:
    n_step: 4  # number of flow steps in scale block in the dequantization module
    n_conv: 0  # additional convolution layers before scale block
  estimators:
    c_u_mult: 6  # number of channels in hidden layers, expressed as multiples of c_u 
    n_conv: 32  # number of gated_convolutions in gated_resnet in parameters estimators in normalizing flow
    attn_red_ratio: 0.5  # 0-1 slider to adjust influence of attention mechanism
  prior_variance: 1.0  # standard deviation of Gaussian prior
  lr: 1.0e-4  # learning rate
  gamma: 0.9  # learning rate decay rate (per epoch)
data:
  dataset: celeb  # sub-folder at data_dir
  data_dir: your_path_here  # path to data
  log_dir: logs  # path to store training and testing logs
training:
  run_name: null  # name of training run. defaults to integer increments
  filename: train  # filename of NPZ file to be used for training
  epochs: 128
  batch_size: 8
  sample_size: null  # sample subset of training data from NPZ. accepts 0-1 float representing proportion or integers representing absolute counts
  log_interval: 4  # log every n-th step
  flush: 16  # flush logs from RAM to txt file every n-th step
testing:
  run_name: null  # name of run to test. defaults to the largest integer filename
  filename: test  # filename of NPZ file to be used for testing
  batch_size: 128
  sample_size: 256  # sample subset of training data from NPZ. accepts 0-1 float representing proportion or integers representing absolute counts  
  posterior_sample_size: 16  # number of samples to generate per prediction
  temperature: [0.0001, 0.5, 0.8, 1.0]  # standard deviation values of Gaussian prior to generate outputs at
```

## Contributing

This repository will likely continue development through the GitHub Issues tab. If you would like 
to contribute or have any inquiries, feel free to create a pull request or drop Joel an email at 
joel_ang@imcb.a-star.edu.sg. Thank you.

