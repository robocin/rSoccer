#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import threading
import time
import traceback
from pprint import pprint

import gym
import rc_gym
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from agents import agentSAC

if __name__ == "__main__":

    play_threads = []
    train_process = None
    mp.set_start_method('spawn')
    collected_samples = mp.Value(ctypes.c_ulonglong, 0)
    finish_event = mp.Event()
    exp_queue = None

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=True,
                            action="store_true", help="Enable cuda")
        parser.add_argument("--name", default='deepvss',
                            help="Experiment name")
        parser.add_argument("--resume", default=[], action='append',
                            nargs='?', help="Pre-trained model to be loaded")
        parser.add_argument("--exp", default=[], action='append',
                            nargs='?', help="Load experience buffer")
        parser.add_argument("--test", default=False,
                            action='store_true', help="Test mode")
        parser.add_argument("--collected", default=-1, type=int,
                            help="The starting collected samples: defaults to zero or the value stored with the pretrained model")
        parser.add_argument("--processed", default=-1, type=int,
                            help="The starting processed samples: defaults to zero or the value stored with the pretrained model")
        parser.add_argument("--params", default='parameters_sac.json',
                            help="Path to a json parameters file")
        parser.add_argument("--env", default='VSSReach-v0', type=str,
                            help='Gym id of the environment, ex.: VSSReach-v0')

        args = parser.parse_args()

        with open(args.params, 'r') as params_file:
            model_params = json.load(params_file)

        print(f' === Running: <{args.name}> experiment ===')
        print('Model params:')
        pprint(model_params)

        env = gym.make(args.env)
        state_space = env.observation_space
        action_space = env.action_space
        env.close()
        print("State shape: " + str(state_space.shape) +
              " Action shape: " + str(action_space.shape))

        model_params['state_shape'] = state_space
        model_params['action_shape'] = action_space

        train = agentSAC.train
        play = agentSAC.play
        create_actor_model = agentSAC.create_actor_model
        load_actor_model = agentSAC.load_actor_model

        device = torch.device("cuda" if args.cuda else "cpu")
        net = create_actor_model(model_params, device)

        checkpoint = {}

        if len(args.resume) > 0:  # load checkpoint
            args.resume = args.resume[0]
            load = True
            if args.resume is None:
                print("Looking for default pth file")
                args.resume = "model/" + args.name + ".pth"
                load = os.path.isfile(args.resume)
                if not load:
                    print('File not found:"%s" (nothing to resume)' %
                          args.resume)

            if load:
                print("=> loading checkpoint '%s'" % args.resume)
                checkpoint = torch.load(args.resume, map_location=device)
                net = load_actor_model(net, checkpoint)

                if args.test:
                    checkpoint['collected_samples'] = checkpoint['processed_samples'] = 0

                if args.processed >= 0:  # override values in checkpoint
                    checkpoint['processed_samples'] = args.processed

                if args.collected >= 0:  # override values in checkpoint
                    checkpoint['collected_samples'] = args.collected

                if 'collected_samples' in checkpoint:
                    collected_samples.value = checkpoint['collected_samples']

                print("=> collected samples: %d, processed_samples: %d" %
                      (collected_samples.value, checkpoint['processed_samples']))

        if len(args.exp) > 0:  # load experience buffer
            checkpoint['exp'] = args.exp[0]

        writer_path = model_params['data_path'] + '/logs/' + args.name
        writer = SummaryWriter(log_dir=writer_path+"/agents", comment="-agent")

        queue_size = model_params['batch_size']
        exp_queue = mp.Queue(maxsize=queue_size)

        th_a = threading.Thread(target=play, args=(model_params, net, device,
                                                   exp_queue, args.env,
                                                   args.test, writer,
                                                   collected_samples,
                                                   finish_event))
        play_threads.append(th_a)
        th_a.start()

        if args.test:
            while True:
                time.sleep(0.01)

        else:
            model_params['run_name'] = args.name
            model_params['writer_path'] = writer_path
            model_params['action_format'] = '2f'
            model_params['state_format'] = '27f'
            net.share_memory()
            train_process = mp.Process(target=train, args=(
                model_params, net, device, exp_queue, finish_event, checkpoint))
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

        finish_event.set()
        if exp_queue:
            while exp_queue.qsize() > 0:
                exp_queue.get()

        print('queue is empty')

        if train_process is not None:
            train_process.join()

        print("Waiting for threads to finish...")
        play_threads = [t.join()
                        for t in play_threads if t is not None and t.isAlive()]
