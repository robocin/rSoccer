import os
import time
from collections import namedtuple
from copy import deepcopy

import gym
import numpy as np
import rc_gym
import torch
import wandb
from lib import common, ddpg_model
from agents.Utils.Buffers import ReplayBuffer

ExperienceFirstLast = namedtuple('ExperienceFirstLast',
                                 ('state', 'action', 'reward', 'last_state'))


def calc_loss_ddpg_critic(batch, crt_net, tgt_act_net, tgt_crt_net,
                          gamma, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    mem_loss = None

    states_v = torch.tensor(states, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.float32)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    done_mask = torch.BoolTensor(dones)

    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    # critic
    q_v = crt_net(states_v, actions_v)
    last_act_v = tgt_act_net(next_states_v)
    q_last_v = tgt_crt_net(next_states_v, last_act_v)
    q_last_v[done_mask] = 0.0
    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma
    critic_loss_v = (q_v - q_ref_v.detach()).pow(2)
    critic_loss_v = critic_loss_v.mean()
    return critic_loss_v, mem_loss


def calc_loss_ddpg_actor(batch, act_net, crt_net, cuda=False, cuda_async=False):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states, dtype=torch.float32)
    next_states_v = torch.tensor(next_states, dtype=torch.float32)
    actions_v = torch.tensor(actions, dtype=torch.float32)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    done_mask = torch.BoolTensor(dones)

    if cuda:
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    # actor
    cur_actions_v = act_net(states_v)
    actor_loss_v = -crt_net(states_v, cur_actions_v)
    actor_loss_v = actor_loss_v.mean()
    return actor_loss_v


def play(params, net, device, exp_queue,
         env_name, test, finish_event, n_episodes):
    try:
        env = gym.make(env_name)
        agent = ddpg_model.AgentDDPG(net, device=device,
                                     ou_teta=params['ou_teta'],
                                     ou_sigma=params['ou_sigma'])
        state = env.reset()
        epi_reward = 0
        then = time.time()
        evaluation = False
        steps = 0

        for matches_played in range(n_episodes):
            if not finish_event.is_set():
                env.close()
                break
            action = agent(state, steps)
            next_state, reward, done, info = env.step(action)
            steps += 1
            epi_reward += reward
            next_state = next_state if not done else None
            exp = ExperienceFirstLast(state, action, reward, next_state)
            state = next_state

            if not test and not evaluation:
                exp_queue.put(exp)
            elif test:
                env.render()

            if done:
                fps = steps/(time.time() - then)
                then = time.time()
                log_dict = deepcopy(info)
                log_dict['fps'] = fps
                wandb.log(log_dict, matches_played)
                print(f'<======Match {matches_played}======>')
                print(f'-------Reward:', epi_reward)
                print(f'-------FPS:', fps)
                print(f'<==================================>\n')
                epi_reward = 0
                steps = 0
                state = env.reset()
                agent.ou_noise.reset()

    except KeyboardInterrupt:
        print("...Agent Finishing...")
        finish_event.set()

    except Exception as e:
        print(e)

    finally:
        if not finish_event.is_set():
            print("Agent set finish flag.")
            finish_event.set()


def train(model_params, act_net, device,
          exp_queue, finish_event, checkpoint=None):

    try:
        run_name = model_params['run_name']
        data_path = model_params['data_path']

        exp_buffer = ReplayBuffer(model_params['replay_size'])

        crt_net = ddpg_model.DDPGCritic(model_params['state_shape'].shape[0],
                                        model_params['action_shape'].shape[0])
        crt_net = crt_net.to(device)
        optimizer_act = torch.optim.Adam(act_net.parameters(),
                                         lr=model_params['learning_rate'])
        optimizer_crt = torch.optim.Adam(crt_net.parameters(),
                                         lr=model_params['learning_rate'])
        tgt_act_net = ddpg_model.AgentDDPG(net, device=device,
                                           ou_teta=params['ou_teta'],
                                           ou_sigma=params['ou_sigma'])
        tgt_crt_net = ddpg_model.DDPGCritic(model_params['state_shape'].shape[0],
                                            model_params['action_shape'].shape[0])

        for target_param, param in zip(tgt_act_net.parameters(),
                                       act_net.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(tgt_crt_net.parameters(),
                                       crt_net.parameters()):
            target_param.data.copy_(param.data)

        act_net.train(True)
        crt_net.train(True)
        tgt_act_net.target_model.train(True)
        tgt_crt_net.target_model.train(True)

        collected_samples = 0
        processed_samples = 0
        best_reward = -np.inf

        if checkpoint is not None:
            if 'state_dict_crt' in checkpoint:
                if 'collected_samples' in checkpoint:
                    collected_samples = checkpoint['collected_samples']

                if 'processed_samples' in checkpoint:
                    processed_samples = checkpoint['processed_samples']

                reward_avg = best_reward = checkpoint['reward']
                crt_net.load_state_dict(checkpoint['state_dict_crt'])
                tgt_act_net.target_model.load_state_dict(
                    checkpoint['tgt_act_state_dict'])
                tgt_crt_net.target_model.load_state_dict(
                    checkpoint['tgt_crt_state_dict'])
                optimizer_act.load_state_dict(checkpoint['optimizer_act'])
                optimizer_crt.load_state_dict(checkpoint['optimizer_crt'])
                print("=> loaded checkpoint '%s' (collected samples: %d, processed_samples: %d, with reward %f)" % (
                    run_name, collected_samples, processed_samples, reward_avg))

            if 'exp' in checkpoint:  # load experience buffer
                exp = checkpoint['exp']
                load = True
                if exp is None:
                    print("Looking for default exb file")
                    exp = data_path + "/buffer/" + run_name + ".exb"
                    load = os.path.isfile(exp)
                    if not load:
                        print('File not found:"%s" (nothing to resume)' % exp)

                if load:
                    print("=> Loading experiences from: " + exp + "...")
                    exp_buffer.load_exps_from_file(exp)
                    print("%d experiences loaded" % (len(exp_buffer)))

        target_net_sync = model_params['target_net_sync']
        replay_initial = model_params['replay_initial']
        next_check_point = processed_samples + \
            model_params['save_model_frequency']
        next_net_sync = processed_samples + model_params['target_net_sync']
        queue_max_size = batch_size = model_params['batch_size']
        writer_path = model_params['writer_path']
        writer = SummaryWriter(log_dir=writer_path+"/train")
        tracker = common.RewardTracker(writer)

        actor_loss = 0.0
        critic_loss = 0.0
        last_loss_average = 0.0

        # training loop:
        print("Training started.")
        while not finish_event.is_set():
            new_samples = 0

            # print("get qsize: %d" % size)
            rewards_gg = [0 for _ in range(0, max(1, int(queue_max_size)))]
            for i in range(0, max(1, int(queue_max_size))):
                exp = exp_queue.get()
                if exp is None:
                    break
                exp_buffer._add(exp)
                rewards_gg[i] = exp.reward
                new_samples += 1

            if len(exp_buffer) < replay_initial:
                continue

            collected_samples += new_samples

            # training loop:
            while exp_queue.qsize() < queue_max_size/2:
                mem_w = None
                if not model_params['per']:
                    batch = exp_buffer.sample(batch_size)
                else:
                    batch, mem_idxs, mem_w = exp_buffer.sample(batch_size)
                optimizer_crt.zero_grad()
                optimizer_act.zero_grad()

                crt_loss_v, mem_loss = calc_loss_ddpg_critic(batch, crt_net, tgt_act_net.target_model, tgt_crt_net.target_model, gamma=model_params['gamma'],
                                                             cuda=(device.type == "cuda"), cuda_async=True, per=model_params['per'], mem_w=mem_w)
                crt_loss_v.backward()
                optimizer_crt.step()
                if model_params['per']:
                    mem_loss = mem_loss.detach().cpu().numpy()[0]
                    exp_buffer.update_priorities(mem_idxs, mem_loss)

                act_loss_v = calc_loss_ddpg_actor(
                    batch, act_net, crt_net, cuda=(device.type == "cuda"),
                    cuda_async=True)
                act_loss_v.backward()
                optimizer_act.step()

                processed_samples += batch_size
                critic_loss += crt_loss_v.item()
                actor_loss += act_loss_v.item()

            # print("|\n")

            # soft sync

            if target_net_sync >= 1:
                if processed_samples >= next_net_sync:
                    next_net_sync = processed_samples + target_net_sync
                    tgt_act_net.sync()
                    tgt_crt_net.sync()
            else:
                tgt_act_net.alpha_sync(alpha=target_net_sync)  # 1 - 1e-3
                tgt_crt_net.alpha_sync(alpha=target_net_sync)

            if processed_samples >= next_check_point:
                next_check_point = processed_samples + \
                    model_params['save_model_frequency']
                reward_avg = avg_rewards(exp_buffer, 1000)

                if reward_avg > best_reward:
                    best_reward = reward_avg
                    is_best = True
                else:
                    is_best = False

                try:
                    print("saving checkpoint with %d/%d collected/processed samples with best reward %f..." %
                          (collected_samples, processed_samples, best_reward))
                    save_checkpoint({
                        'model_type': 'ddpg',
                        'collected_samples': collected_samples,
                        'processed_samples': processed_samples,
                        'state_dict_act': act_net.state_dict(),
                        'state_dict_crt': crt_net.state_dict(),
                        'tgt_act_state_dict': tgt_act_net.target_model.state_dict(),
                        'tgt_crt_state_dict': tgt_crt_net.target_model.state_dict(),
                        'reward': reward_avg,
                        'optimizer_act': optimizer_act.state_dict(),
                        'optimizer_crt': optimizer_crt.state_dict(),
                    }, is_best, "model/" + run_name + ".pth")

                    if processed_samples > last_loss_average:
                        actor_loss = batch_size*actor_loss / \
                            (processed_samples-last_loss_average)
                        critic_loss = batch_size*critic_loss / \
                            (processed_samples-last_loss_average)
                        print("avg_reward:%.4f, avg_loss:%f" %
                              (reward_avg, actor_loss))
                        tracker.track_training(
                            processed_samples, reward_avg, actor_loss, critic_loss)
                        actor_loss = 0.0
                        critic_loss = 0.0
                        last_loss_average = processed_samples

                    exp_buffer.sync_exps_to_file(
                        data_path + "/buffer/" + run_name + ".exb")

                except Exception:
                    with open(run_name + ".err", 'a') as errfile:
                        errfile.write("!!! Exception caught on training !!!")
                        errfile.write(traceback.format_exc())

    except KeyboardInterrupt:
        print("...Training finishing...")

    except Exception:
        print("!!! Exception caught on training !!!")
        print(traceback.format_exc())
        with open(run_name+".err", 'a') as errfile:
            errfile.write("!!! Exception caught on training !!!")
            errfile.write(traceback.format_exc())

    finally:
        if not finish_event.is_set():
            print("Train process set finish flag.")
            finish_event.set()

        print("Training finished.")
