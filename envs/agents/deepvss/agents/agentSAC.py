import os
import shutil
import time
import traceback
from collections import deque

import gym
import numpy as np
import ptan
import rc_gym
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from lib import common, sac_model
from tensorboardX import SummaryWriter

gradMax = 0
gradAvg = 0


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '_best.pth')


def inspectGrads(grad):
    global gradMax, gradAvg
    maxg = grad.max()
    maxg = max(-grad.min(), maxg)
    # print("**** MAX GRAD: %.5f" % maxg + " OLD: %.5f" % gradMax + " AVG: %.5f" % gradAvg + " ****")
    if maxg > gradMax:
        print("**** NEW MAX GRAD: %.5f" % maxg + " OLD: %.5f" %
              gradMax + " AVG: %.5f" % gradAvg + " ****")
        gradMax = maxg
    gradAvg = .1*maxg + .9*gradAvg


def avg_rewards(exp_buffer, total):

    if total > len(exp_buffer):
        total = len(exp_buffer)

    count = 0
    reward = 0
    pos = exp_buffer.pos
    while count < total:
        reward += exp_buffer.buffer[pos].reward
        pos -= 1
        if pos < 0:
            pos = len(exp_buffer)-1
        count += 1

    return reward


def create_actor_model(model_params, device):
    act_net = sac_model.GaussianPolicy(model_params['state_shape'].shape[0]+3,
                                       model_params['action_shape'].shape[0]
                                       ).to(device)
    return act_net


def load_actor_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict_act'])

    return net


def play(params, net, device, exp_queue, env_id, test,
         writer, collected_samples, finish_event):

    try:
        agent_env = gym.make(env_id)
        agent = sac_model.AgentSAC(net, device=device)

        print(f"Started from sample {collected_samples.value}.")
        state, objective = agent_env.reset()
        matches_played = 0
        epi_reward = 0
        then = time.time()
        eval_freq_matches = params['eval_freq_matches']
        evaluation = False
        steps = 0

        while not finish_event.is_set():
            state = np.concatenate([state, objective[:-1]/0.85])
            state = np.concatenate([state, objective[-1:]/360])
            action = agent([state])[0]
            next_state, reward, done, info = agent_env.step(action)
            objective = info['objective']
            info = info['reward_shaping']
            steps += 1
            epi_reward += reward
            next_state = next_state if not done else None
            ns = np.concatenate(
                (next_state, objective[:-1]/0.85)) if not done else None
            ns = np.concatenate(
                (ns, objective[-1:]/360)) if not done else None
            exp = ptan.experience.ExperienceFirstLast(state, action,
                                                      reward, ns)
            state = next_state
            if not test and not evaluation:
                exp_queue.put(exp)
            elif test:
                agent_env.render()

            if done:
                fps = steps/(time.time() - then)
                then = time.time()
                writer.add_scalar("rw/total", epi_reward, matches_played)
                writer.add_scalar("rw/steps_ep", steps, matches_played)
                writer.add_scalar("rw/reach_score",
                                  info['reach_score'],
                                  matches_played)
                writer.add_scalar("rw/move", info['move'], matches_played)
                writer.add_scalar("rw/time", info['time'], matches_played)
                writer.add_scalar("rw/energy", info['energy'], matches_played)
                print(f'<======Match {matches_played}======>')
                print(f'-------Reward:', epi_reward)
                print(f'-------FPS:', fps)
                print(f'<==================================>\n')
                epi_reward = 0
                steps = 0
                matches_played += 1
                state, objective = agent_env.reset()

                if not test and evaluation:  # evaluation just finished
                    writer.add_scalar("eval/rw", epi_reward, matches_played)
                    print("evaluation finished")

                evaluation = matches_played % eval_freq_matches == 0

                if not test and evaluation:  # evaluation just started
                    print("Evaluation started")

            collected_samples.value += 1

    except KeyboardInterrupt:
        print("...Agent Finishing...")
        finish_event.set()

    except Exception:
        print("!!! Exception caught on agent!!!")
        print(traceback.format_exc())

    finally:
        if not finish_event.is_set():
            print("Agent set finish flag.")
            finish_event.set()

    agent_env.close()
    print("Agent finished.")


def train(model_params, act_net, device, exp_queue, finish_event, checkpoint=None):

    try:
        run_name = model_params['run_name']
        data_path = model_params['data_path']

        exp_buffer = common.PersistentExperienceReplayBuffer(experience_source=None,
                                                             buffer_size=model_params['replay_size'])
        exp_buffer.set_state_action_format(
            state_format=model_params['state_format'], action_format=model_params['action_format'])

        crt_net = sac_model.QNetwork(model_params['state_shape'].shape[0]+2,
                                     model_params['action_shape'].shape[0]).to(device)
        optimizer_act = torch.optim.Adam(
            act_net.parameters(), lr=model_params['learning_rate'])
        optimizer_crt = torch.optim.Adam(crt_net.parameters(), lr=model_params['learning_rate'],
                                         weight_decay=model_params['weight_decay'])
        tgt_crt_net = ptan.agent.TargetNet(crt_net)
        if model_params['automatic_entropy_tuning']:
            target_entropy = - \
                torch.prod(torch.Tensor(
                    model_params['action_shape'].shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam(
                [log_alpha], lr=model_params['learning_rate'])

        alpha = model_params['alpha']
        gamma = model_params['gamma']
        sync_freq = model_params['sync_freq']

        act_net.train(True)
        crt_net.train(True)
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
                    print("%d experiences loaded" % (exp_buffer.__len__()))

        target_net_sync = model_params['target_net_sync']
        replay_initial = model_params['replay_initial']
        next_check_point = processed_samples + \
            model_params['save_model_frequency']
        next_net_sync = processed_samples + model_params['target_net_sync']
        queue_max_size = batch_size = model_params['batch_size']
        writer_path = model_params['writer_path']
        writer = SummaryWriter(log_dir=writer_path+"/train")
        tracker = common.RewardTracker(writer)

        policy_loss = 0.0
        qf1_loss = 0.0
        qf2_loss = 0.0
        alpha_loss = 0.0
        last_loss_average = 0.0
        first = True

        # training loop:
        while not finish_event.is_set():
            new_samples = 0

            #print("get qsize: %d" % size)
            for _ in range(0, max(1, int(queue_max_size))):
                exp = exp_queue.get()
                if exp is None:
                    break
                exp_buffer._add(exp)
                new_samples += 1

            if len(exp_buffer) < replay_initial:
                continue

            collected_samples += new_samples

            # training loop:
            while exp_queue.qsize() < queue_max_size/2:
                if first:
                    print("Training started.")
                    print(crt_net)
                    print(act_net)
                    first = False
                batch = exp_buffer.sample(batch_size=batch_size)
                state_batch, action_batch, reward_batch,\
                    mask_batch, next_state_batch = common.unpack_batch(batch)

                if device == torch.device('cuda'):
                    state_batch = torch.FloatTensor(
                        state_batch).cuda(non_blocking=True)
                    next_state_batch = torch.FloatTensor(
                        next_state_batch).cuda(non_blocking=True)
                    action_batch = torch.FloatTensor(
                        action_batch).cuda(non_blocking=True)
                    reward_batch = torch.FloatTensor(
                        reward_batch).cuda(non_blocking=True).unsqueeze(1)
                    mask_batch = torch.BoolTensor(
                        mask_batch).cuda(non_blocking=True)
                else:
                    state_batch = torch.FloatTensor(state_batch)
                    next_state_batch = torch.FloatTensor(next_state_batch)
                    action_batch = torch.FloatTensor(action_batch)
                    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
                    mask_batch = torch.BoolTensor(mask_batch)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = act_net.sample(
                        next_state_batch)
                    qf1_next_target, qf2_next_target = tgt_crt_net.target_model(
                        next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(
                        qf1_next_target,
                        qf2_next_target) - alpha * next_state_log_pi
                    min_qf_next_target[mask_batch] = 0.0
                    next_q_value = reward_batch + gamma * (min_qf_next_target)
                # Two Q-functions to mitigate
                # positive bias in the policy improvement step
                qf1, qf2 = crt_net(state_batch, action_batch)
                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf1_loss = F.mse_loss(qf1, next_q_value)
                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2, next_q_value)

                pi, log_pi, _ = act_net.sample(state_batch)

                qf1_pi, qf2_pi = crt_net(state_batch, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                optimizer_crt.zero_grad()
                qf1_loss.backward()
                optimizer_crt.step()

                optimizer_crt.zero_grad()
                qf2_loss.backward()
                optimizer_crt.step()

                optimizer_act.zero_grad()
                policy_loss.backward()
                optimizer_act.step()

                if model_params['automatic_entropy_tuning']:
                    alpha_loss = -(log_alpha * (log_pi +
                                                target_entropy
                                                ).detach()).mean()

                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                    alpha = log_alpha.exp()
                    alpha_tlogs = alpha.clone()  # For TensorboardX logs
                else:
                    alpha_loss = torch.tensor(0.).to(device)
                    alpha_tlogs = torch.tensor(alpha)  # For TensorboardX logs
                processed_samples += batch_size

            # print("|\n")

            # soft sync
            if sync_freq == 0:
                if target_net_sync >= 1:
                    if processed_samples >= next_net_sync:
                        next_net_sync = processed_samples + target_net_sync
                        tgt_crt_net.sync()
                else:
                    tgt_crt_net.alpha_sync(alpha=target_net_sync)
                sync_freq = model_params['sync_freq']
            else:
                sync_freq -= 1

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
                        'model_type': 'sac',
                        'collected_samples': collected_samples,
                        'processed_samples': processed_samples,
                        'state_dict_act': act_net.state_dict(),
                        'state_dict_crt': crt_net.state_dict(),
                        'tgt_crt_state_dict': tgt_crt_net.target_model.state_dict(),
                        'reward': reward_avg,
                        'optimizer_act': optimizer_act.state_dict(),
                        'optimizer_crt': optimizer_crt.state_dict(),
                    }, is_best, "model/" + run_name + ".pth")

                    if processed_samples > last_loss_average:
                        policy_loss = batch_size*policy_loss / \
                            (processed_samples-last_loss_average)
                        print("avg_reward:%.4f, avg_loss:%f" %
                              (reward_avg, policy_loss))
                        tracker.track_training_sac(
                            processed_samples, reward_avg, policy_loss.item(),
                            qf1_loss.item(), qf2_loss.item(), alpha_tlogs.item(),
                            alpha_loss.item())
                        policy_loss = 0.0
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
