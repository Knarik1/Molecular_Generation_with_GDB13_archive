#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq.cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import ast
import random
import sys
import logging
import functools
import re
import math
from tqdm import tqdm
from rdkit import Chem
import time
import selfies as sf
import parmap

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.utils import build_logger

import importlib

import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb
rkrb.DisableLog('rdApp.error')

if "METASEQ_SERVICE_CONSTANTS_MODULE" not in os.environ:
    constants_module = importlib.import_module("metaseq.service.constants")
else:
    constants_module = importlib.import_module(
        os.environ["METASEQ_SERVICE_CONSTANTS_MODULE"]
    )
TOTAL_WORLD_SIZE = constants_module.TOTAL_WORLD_SIZE
LAUNCH_ARGS = constants_module.LAUNCH_ARGS
INFERENCE_ARG_OVERRIDES = constants_module.INFERENCE_ARG_OVERRIDES

logger = build_logger()


def input_loop():
    inp = []
    while True:
        try:
            # green display, bold user prompt
            display = (
                "\033[32mPrompt (ctrl-D to end input, ctrl-C to quit):\n\033[0;1m"
                if not inp
                else ""
            )
            data = input(display)
            inp.append(data)
        except KeyboardInterrupt:
            # reset the formatting
            sys.stdout.write("\033[0m")
            raise
        except EOFError:
            break
        # reset the formatting
        sys.stdout.write("\033[0m")
    logger.debug(f"Input: {inp}")
    return "\n".join(inp)


def make_caninical_smiles(selfies):
    canon_smiles = None
    smiles = sf.decoder(selfies)

    try:
        canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        pass  

    return canon_smiles


def worker_main(cfg: MetaseqConfig, namespace_args=None):
    global generator
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    # quiet some of the stuff for visual aspects
    logging.getLogger("metaseq.hub_utils").setLevel(logging.WARNING)

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")

    if torch.distributed.is_initialized():
        request_object = distributed_utils.broadcast_object(
            None, src_rank=0, group=distributed_utils.get_global_group()
        )
        
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        batch_size = cfg.dataset.batch_size
        seed = cfg.common.seed
        file_path = cfg.generation.output_file_path

        if os.path.exists(file_path):
            print(f"{file_path} already exists")
            sys.exit(1)

        csv_file = open(file_path, "wt+")
        write_func = functools.partial(csv_file.write)

        prompt = [[]]*batch_size

        request_object = {
            "inputs": prompt,
            "max_tokens": None,
            "min_tokens": [1]*batch_size,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.sampling_topp,
            "logprobs": cfg.generation.logprobs,
            "n": cfg.generation.beam,
            "best_of": None,
            "echo": False,
            "stop": None,
            "seed": seed,
            "use_cuda": True,
            }
        
        print(request_object)

        for i in tqdm(range(math.ceil(cfg.generation.generation_len/batch_size))):
            generations = generator.generate(**request_object)
            smiles_batch = list(map(lambda x: x[0]["text"], generations))

            if cfg.generation.mol_repr == "smiles":
                # write in a file        
                write_func("\n".join(smiles_batch) + "\n")
            else:        
                try:
                    # if selfies
                    canon_smiles_batch = parmap.map(make_caninical_smiles, smiles_batch, pm_processes=6)

                    # write in a file        
                    write_func("\n".join(canon_smiles_batch) + "\n")
                except:
                    pass  

        print(f"Saved in {file_path}.")                        

    else:
        # useful in FSDP setting
        while True:
            request_object = distributed_utils.broadcast_object(
                None, src_rank=0, group=distributed_utils.get_global_group()
            )
            _ = generator.generate(**request_object)

            


def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

    model_overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    model_overrides.update(INFERENCE_ARG_OVERRIDES)
    cfg.common_eval.model_overrides = str(model_overrides)

    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    if os.getenv("SLURM_NODEID") is None:
        logger.warning(
            f"Missing slurm configuration, defaulting to 'use entire node' for API"
        )
        os.environ["SLURM_NODEID"] = "0"
        os.environ["SLURM_NNODES"] = "1"
        os.environ["SLURM_NTASKS"] = "1"
        import socket

        os.environ["SLURM_STEP_NODELIST"] = socket.gethostname()
    cli_main()
