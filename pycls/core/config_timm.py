#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import os

from pycls.core.io import pathmgr
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ------------------------------- EfficientNet options ------------------------------- #
_C.EN = CfgNode()

# Stem width
_C.EN.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EN.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EN.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EN.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) ratio
_C.EN.SE_R = 0.25

# Strides for each stage (applies to the first block of each stage)
_C.EN.STRIDES = []

# Kernel sizes for each stage
_C.EN.KERNELS = []

# Head width
_C.EN.HEAD_W = 1280

# Drop connect ratio
_C.EN.DC_RATIO = 0.0

# Dropout ratio
_C.EN.DROPOUT_RATIO = 0.0

# ---------------------------------- Dataset options ----------------------------------- #
_C.data_dir = '/usr/Downloads/imagenet/'

_C.dataset = 'imagenet'

#'dataset type (default: ImageFolder/ImageTar if empty)'
_C.train_split = 'train'

#'dataset train split (default: train)'
_C.val_split = 'validation'

#'dataset validation split (default: validation)'
_C.dataset_download = False

#'path to class to idx mapping file (default: "")'
_C.class_map = ''

# ---------------------------------- Model options ----------------------------------- #

_C.model = 'efficientnet'

# Start with pretrained version of specified network (if avail)
_C.pretrained = False

# PATH 'Initialize model from this checkpoint (default: none)'
_C.initial_checkpoint = ''

#'prevent resume of optimizer state when resuming model'
_C.resume = ''

_C.no_resume_opt = False

#'prevent resume of optimizer state when resuming model'
_C.num_classes = None

#'Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.'
_C.gp = None

#help='Image patch size (default: None => model default)')
_C.img_size = None

#N 'Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
_C.input_size = 3

#'Input image center crop percent (for validation only)'
_C.crop_pct = None

#'Override mean pixel value of dataset'
_C.mean = None

#'Override std deviation of of dataset'
_C.std = None

#'Image resize interpolation type (overrides model)'
_C.interpolation = ''

#'input batch size for training (default: 128)'
_C.batch_size = 128

#'validation batch size override (default: 128)'
_C.vb = 128

# ---------------------------------- Optimizer options ---------------------------------- #

#'Optimizer (default: "sgd"')
_C.opt = 'sgd'

#'Optimizer Epsilon (default: None, use opt default)'
_C.opt_eps = None

#'Optimizer Betas (default: None, use opt default)'
_C.opt_betas = None

#'Optimizer momentum (default: 0.9)'
_C.momentum = 0.9

#'weight decay (default: 2e-5)'
_C.weight_decay = 2e-5

#'Clip gradient norm (default: None, no clipping)'
_C.clip_grad = None

#'Gradient clipping mode. One of ("norm", "value", "agc")'
_C.clip_mode = 'norm'

# ---------------------------------- Learning rate schedule parameters ---------------------------------- #
#'LR scheduler (default: "step"'
_C.sched = 'cosine'

#'learning rate (default: 0.05)'
_C.lr = 0.05

#'learning rate noise on/off epoch percentages'
_C.lr_noise = None

#'learning rate noise limit percent (default: 0.67)'
_C.lr_noise_pct = 0.67

#'learning rate noise std-dev (default: 1.0)'
_C.lr_noise_std = 1.0

#'learning rate cycle len multiplier (default: 1.0)'
_C.lr_cycle_mul = 1.0

#'amount to decay each learning rate cycle (default: 0.5)
_C.lr_cycle_decay = 0.5

#'learning rate cycle limit, cycles enabled if > 1'
_C.lr_cycle_limit = 1

#'learning rate k-decay for cosine/poly (default: 1.0)'
_C.lr_k_decay = 1.0

#'warmup learning rate (default: 0.0001)'
_C.warmup_lr = 0.0001

#'lower lr bound for cyclic schedulers that hit 0 (1e-5)'
_C.min_lr = 1e-6

#'number of epochs to train (default: 300)'
_C.epochs = 300

#'epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).'
_C.epoch_repeats = 0.

#'manual epoch number (useful on restarts)'
_C.start_epoch = None

#'epoch interval to decay LR'
_C.decay_epochs = 100

#'epochs to warmup LR, if scheduler supports'
_C.warmup_epochs = 3

#'epochs to cooldown LR at min_lr, after cyclic schedule ends'
_C.cooldown_epochs = 10

#'patience epochs for Plateau LR scheduler (default: 10'
_C.patience_epochs = 10

#'LR decay rate (default: 0.1)'
_C.decay_rate = 0.1

# ---------------------------------- Augmentation & regularization parameters ---------------------------------- #
#'Disable all training augmentation, override other train aug args'
_C.no_aug = False

#'Random resize scale (default: 0.08 1.0)'
_C.scale = [0.08, 1.0]

#'Random resize aspect ratio (default: 0.75 1.33)'
_C.ratio = [3./4., 4./3.]

#'Horizontal flip training aug probability'
_C.hflip = 0.5

#'Vertical flip training aug probability'
_C.vflip = 0.

#'Color jitter factor (default: 0.4)'
_C.color_jitter = 0.4

#'Use AutoAugment policy. "v0" or "original". (default: None)'
_C.aa = None

#'Number of augmentation repetitions (distributed training only) (default: 0)'
_C.aug_repeats = 0

#'Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.aug_splits = 0

#'Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.'
_C.jsd_loss = False

#'Enable BCE loss w/ Mixup/CutMix use.'
_C.bce_loss = False

#'Threshold for binarizing softened BCE targets (default: None, disabled)'
_C.bce_target_thresh = None

#'Random erase prob (default: 0.)'
_C.reprob = 0.

#'Random erase mode (default: "pixel")'
_C.remode = 'pixel'

#'Random erase count (default: 1)'
_C.recount = 1

#'Do not random erase first (clean) augmentation split'
_C.resplit = False

#'mixup alpha, mixup enabled if > 0. (default: 0.)'
_C.mixup = 0.0

#'cutmix alpha, cutmix enabled if > 0. (default: 0.)'
_C.cutmix = 0.0

#'cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
_C.cutmix_minmax = None

#'Probability of performing mixup or cutmix when either/both is enabled'
_C.mixup_prob = 1.0

#'Probability of switching to cutmix when both mixup and cutmix enabled'
_C.mixup_switch_prob = 0.5

#'How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
_C.mixup_mode = 'batch'

#'Turn off mixup after this epoch, disabled if 0 (default: 0)'
_C.mixup_off_epoch = 0

#'Label smoothing (default: 0.1)'
_C.smoothing = 0.1

#'Training interpolation (random, bilinear, bicubic default: "random")'
_C.train_interpolation = 'random'

#'Dropout rate (default: 0.)'
_C.drop = 0.0

#'Drop connect rate, DEPRECATED, use drop_path (default: None)'
_C.drop_connect = None

#Drop path rate (default: None)'
_C.drop_path = None

#'Drop block rate (default: None)'
_C.drop_block = None

# -------------------------------  Batch norm parameters (only works with gen_efficientnet based models currently) ------------------------------- #
#'Use Tensorflow BatchNorm defaults for models that support it (default: False)'
_C.bn_tf = False

#'BatchNorm momentum override (if not None)'
_C.bn_momentum = None

#'BatchNorm epsilon override (if not None)'
_C.bn_eps = None

#'Enable NVIDIA Apex or Torch synchronized BatchNorm.'
_C.sync_bn = False

#'Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")'
_C.dist_bn = 'reduce'

#'Enable separate BN layers per augmentation split.'
_C.split_bn = False



# ---------------------------- Model Exponential Moving Average---------------------------- #
#'Enable tracking moving average of model weights'
_C.model_ema = False

#'Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.'
_C.model_ema_force_cpu = False

#'decay factor for model weights moving average (default: 0.9998)'
_C.model_ema_decay = 0.9998


# -------------------------------- Misc -------------------------------- #
#'random seed (default: 42)'
_C.seed = 42

#'worker seed mode (default: all)'
_C.worker_seeding = 'all'

#'how many batches to wait before logging training status'
_C.log_interval = 50

#'how many batches to wait before writing recovery checkpoint'
_C.recovery_interval = 0

#'number of checkpoints to keep (default: 10)'
_C.checkpoint_hist = 10

#'how many training processes to use (default: 4) _j
_C.workers = 4

#'save images of input bathes every log interval for debugging'
_C.save_images = False

#'use NVIDIA Apex AMP or Native AMP for mixed precision training'
_C.amp = False

#'Use NVIDIA Apex AMP mixed precision'
_C.apex_amp = False

#'Use Native Torch AMP mixed precision'
_C.native_amp = False

#'Force broadcast buffers for native DDP to off.'
_C.no_ddp_bb = False

#'Use channels_last memory layout'
_C.channels_last = False

#'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
_C.pin_mem = False

#'disable fast prefetcher'
_C.no_prefetcher = False

# PATH 'path to output folder (default: none, current dir)'
_C.output = ''

#'name of train experiment, name of sub_folder for output'
_C.experiment = ''

#'Best metric (default: "top1"'
_C.eval_metric = 'top1'

#'Test/inference time augmentation (oversampling) factor. 0=None (default: 0)'
_C.tta = 0

_C.local_rank = 0

#'use the multi_epochs_loader to save time at the beginning of every epoch'
_C.use_multi_epochs_loader = False

#'convert model torchscript for inference'
_C.torchscript = False

#'log training and validation metrics to wandb'
_C.log_wandb = False

#configs file path
_C.cfg = ''

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Minibatch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    err_str = "NUM_GPUS must be divisible by or less than MAX_GPUS_PER_NODE"
    num_gpus, max_gpus_per_node = _C.NUM_GPUS, _C.MAX_GPUS_PER_NODE
    assert num_gpus <= max_gpus_per_node or num_gpus % max_gpus_per_node == 0, err_str
    err_str = "Invalid mode {}".format(_C.LAUNCH.MODE)
    assert _C.LAUNCH.MODE in ["local", "submitit_local", "slurm"], err_str


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)