#!/usr/bin/env python3
import argparse
import ctypes
import math
import os
import sys
import threading
import time

# VSS customization:
import traceback
from importlib.machinery import SourceFileLoader

import gym
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import parameters_sac

import rc_gym
import numpy as np

#  Global variables
writer = None
collected_samples = None


def dict_to_str(dct):
    dict_str = ""
    for key, value in dct.items():
        dict_str += "{}: {}\n".format(key, value)

    return dict_str


if __name__ == "__main__":
    env = None
    play_threads = []
    train_process = None
    mp.set_start_method("spawn")
    collected_samples = mp.Value(ctypes.c_ulonglong, 0)
    finish_event = mp.Event()
    exp_queue = None

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cuda", default=False, action="store_true", help="Enable cuda"
        )
        parser.add_argument("--name", default="Lambada",
                            help="Experiment name")
        parser.add_argument(
            "--resume",
            default=[],
            action="append",
            nargs="?",
            help="Pre-trained model to be loaded",
        )
        parser.add_argument(
            "--exp",
            default=[],
            action="append",
            nargs="?",
            help="Load experience buffer",
        )
        parser.add_argument(
            "--test", default=False, action="store_true", help="Test mode"
        )
        parser.add_argument(
            "--collected",
            default=-1,
            type=int,
            help="The starting collected samples: defaults to zero or the value stored with the pretrained model",
        )
        parser.add_argument(
            "--processed",
            default=-1,
            type=int,
            help="The starting processed samples: defaults to zero or the value stored with the pretrained model",
        )
        parser.add_argument(
            "--real",
            default=False,
            action="store_true",
            help="set env to vss_real_soccer",
        )
        parser.add_argument(
            "--params", default=None, help="Path to a python parameters file"
        )

        args = parser.parse_args()

        if args.real:
            args.simulator = "real"

        model_params = parameters_sac.MODEL_HYPERPARAMS

        if args.params is not None:
            imported_parameters = SourceFileLoader(
                "module.name", args.params
            ).load_module()
            model_params = imported_parameters.MODEL_HYPERPARAMS

        run_name = str(args.name)

        if args.test:
            model_params["epsilon_high_start"] = model_params[
                "epsilon_high_final"
            ] = model_params["epsilon_low"] = 0.02

        sys.stdout.flush()

        env_name = "SSLGoToBallIR-v1"
        env = gym.make(env_name)
        state_shape, action_shape = env.observation_space, env.action_space
        model_params["state_shape"] = state_shape
        model_params["action_shape"] = action_shape

        print(f" === Running: <{run_name}> experiment ===")
        print(f"Model params:\n{dict_to_str(model_params)}\n")

        if model_params["agent"] == "DDPG":
            from agents.agentDDPG import (
                train,
                play,
                create_actor_model,
                load_actor_model,
            )
        elif model_params["agent"] == "SAC":
            from agents.agentSAC import (
                train,
                play,
                create_actor_model,
                load_actor_model,
            )

        device = torch.device("cuda" if args.cuda else "cpu")
        net = create_actor_model(
            model_params, state_shape, action_shape, device
        )

        checkpoint = {}

        if len(args.resume) > 0:  # load checkpoint
            args.resume = args.resume[0]
            load = True
            if args.resume is None:
                print("Looking for default pth file")
                args.resume = "model/" + run_name + ".pth"
                load = os.path.isfile(args.resume)
                if not load:
                    print(
                        'File not found:"%s" (nothing to resume)' % args.resume
                    )

            if load:
                print("=> loading checkpoint '%s'" % args.resume)
                checkpoint = torch.load(args.resume, map_location=device)
                net = load_actor_model(net, checkpoint)

                if args.test:
                    checkpoint["collected_samples"] = checkpoint[
                        "processed_samples"
                    ] = 0

                if args.processed >= 0:  # override values in checkpoint
                    checkpoint["processed_samples"] = args.processed

                if args.collected >= 0:  # override values in checkpoint
                    checkpoint["collected_samples"] = args.collected

                if "collected_samples" in checkpoint:
                    collected_samples.value = checkpoint["collected_samples"]

                print(
                    "=> collected samples: %d, processed_samples: %d"
                    % (collected_samples.value, checkpoint["processed_samples"])
                )

        if len(args.exp) > 0:  # load experience buffer
            checkpoint["exp"] = args.exp[0]

        writer_path = model_params["data_path"] + "/logs/" + run_name
        

        queue_size = model_params["batch_size"]
        exp_queue = mp.Queue(maxsize=queue_size)

        torch.set_num_threads(20)
        print("Threads available: %d" % torch.get_num_threads())

        for i in range(4):
            th_a = mp.Process(
                target=play,
                args=(
                    model_params,
                    net,
                    device,
                    exp_queue,
                    env_name,
                    args.test,
                    collected_samples,
                    finish_event,
                    i
                ),
            )
            play_threads.append(th_a)
            th_a.start()

        if args.test:
            while True:
                time.sleep(0.01)

        else:  # crate train process:
            model_params["run_name"] = run_name
            model_params["writer_path"] = writer_path
            model_params["action_format"] = "3f"
            model_params["state_format"] = f"{state_shape.shape[0]}f"
            net.share_memory()
            train_process = mp.Process(
                target=train,
                args=(
                    model_params,
                    net,
                    device,
                    exp_queue,
                    finish_event,
                    checkpoint,
                ),
            )
            train_process.start()
            train_process.join()
            print("Train process joined.")

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    except Exception:
        print("!!! Exception caught on main !!!")
        print(traceback.format_exc())

    finally:
        if exp_queue:
            while exp_queue.qsize() > 0:
                exp_queue.get()

        print("queue is empty")

        if train_process is not None:
            train_process.join()

        print("Waiting for threads to finish...")
        for p in play_threads:
            p.terminate()
            p.join()
