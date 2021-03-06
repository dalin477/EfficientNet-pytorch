#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model scaler for scaling strategies in https://arxiv.org/abs/2103.06877."""

from math import isclose
import os
import pycls.models.regnet as regnet
from pycls.core.config import cfg
from pycls.models.blocks import adjust_block_compatibility
from pycls.core.io import pathmgr



def scaling_factors(scale_type, scale_factor):
    """
    Computes model scaling factors to allow for scaling along d, w, g, r.

    Compute scaling factors such that d * w * w * r * r == scale_factor.
    Here d is depth, w is width, g is groups, and r is resolution.
    Note that scaling along g is handled in a special manner (see paper or code).

    Examples of scale_type include "d", "dw", "d1_w2", and "d1_w2_g2_r0".
    A scale_type of the form "dw" is equivalent to "d1_w1_g0_r0". The scalar value
    after each scaling dimensions gives the relative scaling along that dimension.
    For example, "d1_w2" indicates to scale twice more along width than depth.
    Finally, scale_factor indicates the absolute amount of scaling.

    The "fast compound scaling" strategy from the paper is specified via "d1_w8_g8_r1".
    """
    if all(s in "dwgr" for s in scale_type):  #判断scale_type中是否是dwgr这种表达放缩；放缩比例默认为1.0（如果放缩的那几个维度比例相同，可以直接用dw这种表达放缩）
        weights = {s: 1.0 if s in scale_type else 0.0 for s in "dwgr"} #判断哪个维度放缩，放缩比例默认为1.0；如{'d': 1.0, 'w': 0.0, 'g': 0.0, 'r': 0.0}
    else:
        weights = {sw[0]: float(sw[1::]) for sw in scale_type.split("_")} #d1_w2这类型表达方式，获取每个维度的放缩比例，如{'d': 1.0, 'w': 2.0}
        weights = {**{s: 0.0 for s in "dwgr"}, **weights}  #其他维度系数为0；{'d': 1.0, 'w': 2.0, 'g': 0.0, 'r': 0.0}
        assert all(s in "dwgr" for s in weights.keys()), scale_type
    sum_weights = weights["d"] + weights["w"] + weights["r"] or weights["g"] / 2 or 1.0  #放缩比例的求和
    #根据放缩倍数scale_factor，以及每个维度的放缩比例，获取每个维度的放缩系数
    d = scale_factor ** (weights["d"] / sum_weights)
    w = scale_factor ** (weights["w"] / sum_weights / 2.0)
    g = scale_factor ** (weights["g"] / sum_weights / 2.0)
    r = scale_factor ** (weights["r"] / sum_weights / 2.0)
    s_actual = d * w * w * r * r
    assert d == w == r == 1.0 or isclose(s_actual, scale_factor, rel_tol=0.01)
    return d, w, g, r

def scale_status(e):

    return d,w,r

def scale_model():
    """
    Scale model blocks by the specified type and amount (note: alters global cfg).

    Scale a model using scaling strategies from "Fast and Accurate Model Scaling".
    For reference on scaling strategies, see: https://arxiv.org/abs/2103.06877.
    For example usage, see GETTING_STARTED, MODEL SCALING section.

    The actual scaling is specified by MODEL.SCALING_TYPE and MODEL.SCALING_FACTOR.
    For example, SCALING_TYPE of "d1_w8_g8_r1" is fast compound scaling and is the
    likely best default option, and SCALING_FACTOR indicates the scaling amount.
    For further details on controlling the scaling, see comments for scaling_factors().

    Note that the scaler must be employed on a standalone config outside of the main
    training loop. This is because it alters the global config, which is typically
    frozen during training. So one should use this function to generate a new config and
    save it to a file, and then evoke training separately on the new config.
    """
    assert cfg.MODEL.TYPE in ["anynet", "effnet", "regnet"]
    # Get scaling factors
    scale_type, scale = cfg.SCALE.SCALING_TYPE, cfg.SCALE.SCALING_FACTOR  #放缩维度及比例，放缩倍数
    d_scale, w_scale, g_scale, r_scale = scaling_factors(scale_type, scale)#获得每个维度的放缩系数
    # scale_name = cfg.MODEL.SCALING_NAME + '_' + scale_type
    cfg.OUT_DIR = os.path.join(cfg.OUT_DIR, cfg.SCALE.SCALING_NAME + '-d' + str(round(d_scale,3)) + '_w' + str(round(w_scale,3)) + '_r' + str(round(r_scale,3)))
    cfg.OPTIM.MAX_EPOCH = cfg.SCALE.MAX_EPOCH
    if not pathmgr.exists(cfg.OUT_DIR):
        pathmgr.mkdirs(cfg.OUT_DIR)
    if cfg.MODEL.TYPE == "regnet":
        # Convert a RegNet to an AnyNet prior to scaling
        regnet.regnet_cfg_to_anynet_cfg()
    if cfg.MODEL.TYPE == "anynet":
        # Scale AnyNet
        an = cfg.ANYNET
        ds, ws, bs, gs = an.DEPTHS, an.WIDTHS, an.BOT_MULS, an.GROUP_WS
        bs = bs if bs else [1] * len(ds)
        gs = gs if gs else [1] * len(ds)
        ds = [max(1, round(d * d_scale)) for d in ds]
        ws = [max(1, round(w * w_scale / 8)) * 8 for w in ws]
        gs = [max(1, round(g * g_scale)) for g in gs]
        gs = [g if g <= 2 else 4 if g <= 5 else round(g / 8) * 8 for g in gs]
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
        an.DEPTHS, an.WIDTHS, an.BOT_MULS, an.GROUP_WS = ds, ws, bs, gs
    elif cfg.MODEL.TYPE == "effnet":
        # Scale EfficientNet
        en = cfg.EN
        ds, ws, bs, sw, hw = en.DEPTHS, en.WIDTHS, en.EXP_RATIOS, en.STEM_W, en.HEAD_W
        ds = [max(1, round(d * d_scale)) for d in ds]
        ws = [max(1, round(w * w_scale / 8)) * 8 for w in ws]
        sw = max(1, round(sw * w_scale / 8)) * 8
        hw = max(1, round(hw * w_scale / 8)) * 8
        ws, bs, _ = adjust_block_compatibility(ws, bs, [1] * len(ds))
        en.DEPTHS, en.WIDTHS, en.EXP_RATIOS, en.STEM_W, en.HEAD_W = ds, ws, bs, sw, hw
    # Scale image resolution
    cfg.TRAIN.IM_SIZE = round(cfg.TRAIN.IM_SIZE * r_scale / 4) * 4
    cfg.TEST.IM_SIZE = round(cfg.TEST.IM_SIZE * r_scale / 4) * 4
