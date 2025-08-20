import argparse
import os
import sys
import os.path as osp
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.util.train_util import AverageMeter
from tinynn.graph.tracer import model_tracer
from tinynn.graph.quantization.algorithm.cross_layer_equalization import (
    cross_layer_equalize,
)
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.converter import TFLiteConverter
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.device import get_device

from mmengine import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="quantizer a model")
    parser.add_argument("config", help="quantization config file path")
    parser.add_argument("model", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to evaluate inference results"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=dict(epochs=5),
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config, modified_constant=args.cfg_options)
    cfg.launcher = args.launcher


if __name__ == "__main__":
    main()