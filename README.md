# FUSE-Flow

## Multi-scale tissue fluorescence mapping with fibre optic ultraviolet excitation and generative modelling
[[Optica](https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-5-673&id=550018)],
[[bioRxiv](https://www.biorxiv.org/content/10.1101/2022.12.28.521919v3)],
[[Liang Research Group](https://liangresearch.com/)]

Joel Lang Yi Ang*, Ko Hui Tan*, Alexander Si Kai Yong*, Chiyo Wan Xuan Tan, Jessica Sze Jia Kng, 
Cyrus Jia Jun Tan, Rachael Hui Kie Soh, Julian Yi Hong Tan, Kaicheng Liang

*Equal contribution

### Abstract

> Cellular imaging of thick samples requires physical sectioning or laser scanning microscopy,
which can be restrictive, involved, and generally incompatible with high-throughput requirements.
We developed fiber optic microscopy with ultraviolet (UV) surface excitation (FUSE), a portable
and quantitative fluorescence imaging platform for thick tissue that enabled quick sub-cellular
imaging without thin sections. We substantially advanced prior UV excitation approaches with
illumination engineering and computational methods. Optical fibers delivered <300nm light with
directional control, enabling unprecedented 50× widefield imaging on thick tissue with sub-nuclear
clarity, and 3D topography of surface microstructure. Probabilistic modeling of high-magnification
images using our normalizing flow architecture FUSE-Flow (made freely available as open-source
software) enhanced low-magnification imaging with measurable localized uncertainty via variational
inference. Comprehensive validation comprised multi-scale fluorescence histology compared with
standard H&E histology, and quantitative analyses of senescence, antibiotic toxicity, and nuclear
DNA content in tissue models via efficient sampling of thick slices from entire murine organs up
to 0.4×8×12mm and 1.3 million cells per surface. This technology addresses long-standing
laboratory gaps in high-throughput studies for rapid cellular insights.

### BibTex

```
@article{Ang:24,
author = {Joel Lang Yi Ang and Ko Hui Tan and Alexander Si Kai Yong and Chiyo Wan XuanTan and Jessica Sze Jia Kng and Cyrus Jia Jun Tan and Rachael Hui Kie Soh and Julian YiHong Tan and Kaicheng Liang},
journal = {Optica},
keywords = {Biomedical imaging; Fiber optic couplers; Optical coherence tomography; Phaseimaging; Scanning microscopy; Spatial resolution},
number = {5},
pages = {673--685},
publisher = {Optica Publishing Group},
title = {Multi-scale tissue fluorescence mapping with fiber optic ultraviolet excitationand generative modeling},
volume = {11},
month = {May},
year = {2024},
url = {https://opg.optica.org/optica/abstract.cfm?URI=optica-11-5-673},
doi = {10.1364/OPTICA.515501},
abstract = {Cellular imaging of thick samples requires physical sectioning or laserscanning microscopy, which can be restrictive, involved, and generally incompatible withhigh-throughput requirements. We developed fiber optic microscopy with ultraviolet (UV)surface excitation (FUSE), a portable and quantitative fluorescence imaging platform forthick tissue that enabled quick sub-cellular imaging without thin sections. Wesubstantially advanced prior UV excitation approaches with illumination engineering andcomputational methods. Optical fibers delivered \&lt;300n            m light withdirectional control, enabling unprecedented 50{\texttimes} widefield imaging on thicktissue with sub-nuclear clarity, and 3D topography of surface microstructure.Probabilistic modeling of high-magnification images using our normalizing flowarchitecture FUSE-Flow (made freely available as open-source software) enhancedlow-magnification imaging with measurable localized uncertainty via variationalinference. Comprehensive validation comprised multi-scale fluorescence histology comparedwith standard H\&E histology, and quantitative analyses of senescence, antibiotictoxicity, and nuclear DNA content in tissue models via efficient sampling of thick slicesfrom entire murine organs up to 0.4{\texttimes}8{\texttimes}12m            m and 1.3million cells per surface. This technology addresses long-standing laboratory gaps inhigh-throughput studies for rapid cellular insights.}
}
```

## Model

<img src="docs/supp_3a.png" width="1000px"></img>

>- Normalising flow module
>  - Scale Block
>    - Squeeze Step
>    - Transition Step
>      - Activation Normalization
>      - Invertible 1x1 Convolution
>    - Conditional Flow Step
>      - Activation Normalization
>      - Invertible 1x1 Convolution
>      - Conditional Affine Injector
>      - Conditional Affine Coupling
>    - Split Step
>- Adaptive Gated U-Net
>  - Gated ResNet
>  - Squeeze-and-Excitation
>  - Concat ELU
>  - Layer Normalization
>- Variational Dequantization Module
>  - Variational perturbation

## Results

<img src="docs/fig_2.png" width="1000px"></img>

>a. Performance overview on fluorescent histological images (held-out) of fresh mouse kidney slice. 
> FUSE-Flow performed domain alignment of input 4X images to reference images in colour and detail 
> while preserving input's coarser features like nuclei positioning and tissue texture.\
>b. FUSE-Flow enhanced nuclear margin sharpness and increased contrast between nuclei and cytoplasm.\
>c. 10X images displayed clear bias in upper right corner due to non-uniform illumination. Bias was 
> absent in model-enhanced images as evidenced by intensity maps correctly corresponding to tissue 
> features.\
>d. FUSE-Flow outputs show no out-of-focus areas, typically seen in higher-magnification images due 
> to tissue regions falling outside objective depth-of-field.\
>e. Multiple samples (n=64) drawn from learnt posterior distribution could estimate conditional 
> standard error to identify regions with highly aleatoric uncertainty. σ=5 
> (or p-value=3e<sup>-7</sup>) was highlighted.

<img src="docs/supp_9b.png" width="1000px"></img>
<img src="docs/supp_10.png" width="1000px"></img>

>- Ablation study on CelebA dataset.

## Getting Started

### _configurations.yaml_

Hyperparameters, architecture modifications, and configurations for training and testing are defined 
in a `configurations.yaml` text file. The file is `.gitignore`'ed for commit cleanliness, so you'll 
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
  epochs: 128  # number of training cycles through entire dataset
  batch_size: 8  # number of data points per training iteration
  sample_size: null  # sample subset of training data from NPZ. accepts 0-1 float representing proportion or integers representing absolute counts
  log_interval: 4  # log every n-th step
  flush: 16  # flush logs from RAM to txt file every n-th step
testing:
  run_name: null  # name of run to test. defaults to the largest integer filename
  filename: test  # filename of NPZ file to be used for testing
  batch_size: 128  # number of data points per testing iteration
  sample_size: 256  # sample subset of training data from NPZ. accepts 0-1 float representing proportion or integers representing absolute counts  
  posterior_sample_size: 16  # number of samples to generate per prediction
  temperature: [0.0001, 0.5, 0.8, 1.0]  # standard deviation values of Gaussian prior to generate outputs at
```

### Data

We will provide datasets for demo purposes soon.

### Training and testing

    .
    ├── .github/              # Linter
    ├── data/                 # Locally downloaded data (optional)
    │   ├── celeb/            # CelebA dataset
    │   ├── fuse/             # FUSE microscopy dataset
    │   └── ...
    ├── data_modules/         # Ancillary data modules
    ├── docs/                 # Images for README
    ├── FUSE_Flow/            # FUSE-Flow model source code
    ├── utils/                # Utility functions to faciliate training and testing
    ├── .gitignore
    ├── configurations.yaml   # Model configurations
    ├── LICENSE
    ├── README.md
    ├── test.py               # Testing script
    └── train.py              # Training script

<img src="docs/file_structure.png" width="1000px"></img>

`train.py` and `test.py` can simply be run once `configuration.yaml` is implemented.

### Pre-training

For best results, pre-training of the SR block is recommended. To do so, follow these steps:
1. Set `no_gan` in `configurations.yaml` to `true`. This will deactivate the other GAN modules and
only train using the loss specified in `edsr_loss` in `configurations.yaml`.
2. Click run on `train.py` after ensuring the rest of `configurations.yaml` are set according to
your specifications. Batch size should likely be much greater since training on just the SR block
requires much less memory.
3. After training is complete, navigate to the relevant log folder containing the results of the
training run. In the `checkpoints` folder, you'll find the latest version of the weights saved.
Copy the weight file.
4. Navigate to `FUSE-Flow/models/pretrain_unet/`. Paste the weight file into this folder. Rename
it to `weights.pth`.
5. Now set `no_gan` in `configurations.yaml` to `false` and ensure that `no_pretrain` is set to
`false` as well. Batch size should likely be reduced now that the other GAN modules are included.
6. Click run on `train.py`.

## Inquiries

If you have any inquiries, feel free to drop an email to Kaicheng (principal investigator and
corresponding author) at liang_kaicheng@imcb.a-star.edu.sg or Joel (lead developer and first
author) at joelangly@outlook.com. Thank you.

## Acknowledgments

We would like to thank Dr. Chee Bing Ong for his invaluable support and pathology-related
consultations, Dr. Nazihah Husna Abdul Aziz for her input into various experiment designs,
Zhe Li Ha for her artistic contributions to figure design, Li Qin Shen for her contributions
to software development for microscope automation, Stefanie Zi En Lim and Jeremy Rui Quan Lee
for their contributions to data preparation, Rachel Yixuan Tan for her contributions to initial
mechanical designs, and Dr. Zesheng Zheng for engineering-related consultations. We would also
like to acknowledge Nikon Imaging Centre (NIC) @ Singapore Bioimaging Consortium (SBIC) for
permitting access to their confocal and brightfield imaging systems. Part of this effort was
conducted at the Institute of Bioengineering & Bioimaging (IBB), A*STAR.
