import os
import shutil
import traceback
import time
import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from lib import common, sac_model

gradMax = 0
gradAvg = 0

#! is_yellow presente na função play do sac
# ? nao sabiamos se era pra remover ou nao, entao deixamos


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + "_best.pth")


def inspectGrads(grad):
    global gradMax, gradAvg
    maxg = grad.max()
    maxg = max(-grad.min(), maxg)
    # print("**** MAX GRAD: %.5f" % maxg + " OLD: %.5f" % gradMax + " AVG: %.5f" % gradAvg + " ****")
    if maxg > gradMax:
        print(
            "**** NEW MAX GRAD: %.5f" % maxg
            + " OLD: %.5f" % gradMax
            + " AVG: %.5f" % gradAvg
            + " ****"
        )
        gradMax = maxg
    gradAvg = 0.1 * maxg + 0.9 * gradAvg


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
            pos = len(exp_buffer) - 1
        count += 1

    return reward


def calc_loss_sac_actor(min_qf_pi, log_pi, model_params):
    alpha = model_params["alpha"]

    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()
    return policy_loss


def calc_loss_sac_critic(
    model_params,
    batch,
    crt_net,
    act_net,
    tgt_crt_net,
    cuda=False,
    cuda_async=False,
):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    (
        state_batch,
        action_batch,
        reward_batch,
        mask_batch,
        next_state_batch,
    ) = common.unpack_batch(batch)

    alpha = model_params["alpha"]
    gamma = model_params["gamma"]

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
    else:
        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mask_batch = torch.BoolTensor(mask_batch)

    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = act_net.sample(
            next_state_batch
        )
        qf1_next_target, qf2_next_target = tgt_crt_net.target_model(
            next_state_batch, next_state_action
        )
        min_qf_next_target = (
            torch.min(qf1_next_target, qf2_next_target)
            - alpha * next_state_log_pi
        )
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

    return qf1_loss, qf2_loss, log_pi, min_qf_pi


def create_actor_model(model_params, state_shape, action_shape, device):
    act_net = sac_model.GaussianPolicy(
        model_params["state_shape"].shape[0],
        model_params["action_shape"].shape[0],
    ).to(device)

    return act_net


def load_actor_model(net, checkpoint):
    net.load_state_dict(checkpoint["state_dict_act"])

    return net


def play(
    params,
    net,
    device,
    exp_queue,
    agent_env,
    test,
    writer,
    collected_samples,
    finish_event,
):

    try:
        agent = sac_model.AgentSAC(
            net,
            device=device,
            ou_teta=params["ou_teta"],
            ou_sigma=params["ou_sigma"],
        )
        print(f"Started from sample {collected_samples.value}.")
        state = agent_env.reset()
        matches_played = 0
        epi_reward = 0
        then = time.time()
        eval_freq_matches = params["eval_freq_matches"]
        evaluation = False
        steps = 0

        while not finish_event.is_set():
            action = agent([state], steps)
            next_state, reward, done, info = agent_env.step(action)
            steps += 1
            epi_reward += reward
            next_state = next_state if not done else None
            exp = ptan.experience.ExperienceFirstLast(
                state, action, reward, next_state
            )
            state = next_state
            if not test and not evaluation:
                exp_queue.put(exp)
            elif test:
                agent_env.render()

            if done:
                fps = steps / (time.time() - then)
                then = time.time()
                writer.add_scalar("rw/total", epi_reward, matches_played)
                writer.add_scalar("rw/steps_ep", steps, matches_played)
                writer.add_scalar(
                    "rw/goal_score", info["goal_score"], matches_played
                )
                writer.add_scalar("rw/move", info["move"], matches_played)
                writer.add_scalar(
                    "rw/ball_grad", info["ball_grad"], matches_played
                )
                writer.add_scalar("rw/energy", info["energy"], matches_played)
                writer.add_scalar(
                    "rw/goals_blue", info["goals_blue"], matches_played
                )
                writer.add_scalar(
                    "rw/goals_yellow", info["goals_yellow"], matches_played
                )
                print(f"<======Match {matches_played}======>")
                print(f"-------Reward:", epi_reward)
                print(f"-------FPS:", fps)
                print(f"<==================================>\n")
                epi_reward = 0
                steps = 0
                matches_played += 1
                state = agent_env.reset()
                agent.ou_noise.reset()

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


def train(
    model_params, act_net, device, exp_queue, finish_event, checkpoint=None
):

    try:
        run_name = model_params["run_name"]
        data_path = model_params["data_path"]

        exp_buffer = common.PersistentExperienceReplayBuffer(
            experience_source=None, buffer_size=model_params["replay_size"]
        )
        exp_buffer.set_state_action_format(
            state_format=model_params["state_format"],
            action_format=model_params["action_format"],
        )

        crt_net = sac_model.QNetwork(
            model_params["state_shape"].shape[0],
            model_params["action_shape"].shape[0],
        ).to(device)
        optimizer_act = torch.optim.Adam(
            act_net.parameters(), lr=model_params["learning_rate"]
        )
        optimizer_crt = torch.optim.Adam(
            crt_net.parameters(),
            lr=model_params["learning_rate"],
            weight_decay=model_params["weight_decay"],
        )
        tgt_crt_net = ptan.agent.TargetNet(crt_net)
        if model_params["automatic_entropy_tuning"]:
            target_entropy = -torch.prod(
                torch.Tensor(model_params["action_shape"].shape).to(device)
            ).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam(
                [log_alpha], lr=model_params["learning_rate"]
            )

        alpha = model_params["alpha"]
        gamma = model_params["gamma"]
        sync_freq = model_params["sync_freq"]

        act_net.train(True)
        crt_net.train(True)
        tgt_crt_net.target_model.train(True)

        collected_samples = 0
        processed_samples = 0
        best_reward = -np.inf

        if checkpoint is not None:
            if "state_dict_crt" in checkpoint:
                if "collected_samples" in checkpoint:
                    collected_samples = checkpoint["collected_samples"]

                if "processed_samples" in checkpoint:
                    processed_samples = checkpoint["processed_samples"]

                reward_avg = best_reward = checkpoint["reward"]
                crt_net.load_state_dict(checkpoint["state_dict_crt"])
                tgt_crt_net.target_model.load_state_dict(
                    checkpoint["tgt_crt_state_dict"]
                )
                optimizer_act.load_state_dict(checkpoint["optimizer_act"])
                optimizer_crt.load_state_dict(checkpoint["optimizer_crt"])
                print(
                    "=> loaded checkpoint '%s' (collected samples: %d, processed_samples: %d, with reward %f)"
                    % (
                        run_name,
                        collected_samples,
                        processed_samples,
                        reward_avg,
                    )
                )

            if "exp" in checkpoint:  # load experience buffer
                exp = checkpoint["exp"]
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

        target_net_sync = model_params["target_net_sync"]
        replay_initial = model_params["replay_initial"]
        next_check_point = (
            processed_samples + model_params["save_model_frequency"]
        )
        next_net_sync = processed_samples + model_params["target_net_sync"]
        queue_max_size = batch_size = model_params["batch_size"]
        writer_path = model_params["writer_path"]
        writer = SummaryWriter(log_dir=writer_path + "/train")
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

            # print("get qsize: %d" % size)
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
            while exp_queue.qsize() < queue_max_size / 2:
                if first:
                    print("Training started.")
                    print(crt_net)
                    print(act_net)
                    first = False

                batch = exp_buffer.sample(batch_size=batch_size)

                qf1_loss, qf2_loss, log_pi, min_qf_pi = calc_loss_sac_critic(
                    model_params,
                    batch,
                    crt_net,
                    act_net,
                    tgt_crt_net.target_model,
                    cuda=(device.type == "cuda"),
                    cuda_async=True,
                )

                optimizer_crt.zero_grad()
                qf1_loss.backward()
                optimizer_crt.step()

                optimizer_crt.zero_grad()
                qf2_loss.backward()
                optimizer_crt.step()

                if model_params["automatic_entropy_tuning"]:
                    alpha_loss = -(
                        log_alpha * (log_pi + target_entropy).detach()
                    ).mean()

                    alpha_optim.zero_grad()
                    alpha_loss.backward()
                    alpha_optim.step()

                    alpha = log_alpha.exp()
                    alpha_tlogs = alpha.clone()  # For TensorboardX logs
                else:
                    alpha_loss = torch.tensor(0.0).to(device)
                    alpha_tlogs = torch.tensor(alpha)  # For TensorboardX logs
                processed_samples += batch_size

                policy_loss = calc_loss_sac_actor(
                    min_qf_pi, log_pi, model_params
                )
                optimizer_act.zero_grad()
                policy_loss.backward()
                optimizer_act.step()
            if processed_samples >= next_check_point:
                next_check_point = (
                    processed_samples + model_params["save_model_frequency"]
                )
                reward_avg = avg_rewards(exp_buffer, 1000)

                if reward_avg > best_reward:
                    best_reward = reward_avg
                    is_best = True
                else:
                    is_best = False

                try:
                    print(
                        "saving checkpoint with %d/%d collected/processed samples with best reward %f..."
                        % (collected_samples, processed_samples, best_reward)
                    )
                    save_checkpoint(
                        {
                            "model_type": "sac",
                            "collected_samples": collected_samples,
                            "processed_samples": processed_samples,
                            "state_dict_act": act_net.state_dict(),
                            "state_dict_crt": crt_net.state_dict(),
                            "tgt_crt_state_dict": tgt_crt_net.target_model.state_dict(),
                            "reward": reward_avg,
                            "optimizer_act": optimizer_act.state_dict(),
                            "optimizer_crt": optimizer_crt.state_dict(),
                        },
                        is_best,
                        "model/" + run_name + ".pth",
                    )
                    if sync_freq == 0:
                        if target_net_sync >= 1:
                            if processed_samples >= next_net_sync:
                                next_net_sync = (
                                    processed_samples + target_net_sync
                                )
                                tgt_crt_net.sync()
                        else:
                            tgt_crt_net.alpha_sync(alpha=target_net_sync)
                        sync_freq = model_params["sync_freq"]
                    else:
                        sync_freq -= 1
                    if processed_samples > last_loss_average:
                        policy_loss = (
                            batch_size
                            * policy_loss
                            / (processed_samples - last_loss_average)
                        )
                        print(
                            "avg_reward:%.4f, avg_loss:%f"
                            % (reward_avg, policy_loss)
                        )
                        tracker.track_training_sac(
                            processed_samples,
                            reward_avg,
                            policy_loss.item(),
                            qf1_loss.item(),
                            qf2_loss.item(),
                            alpha_tlogs.item(),
                            alpha_loss.item(),
                        )
                        policy_loss = 0.0
                        last_loss_average = processed_samples

                        exp_buffer.sync_exps_to_file(
                            data_path + "/buffer/" + run_name + ".exb"
                        )
                except Exception:
                    with open(run_name + ".err", "a") as errfile:
                        errfile.write("!!! Exception caught on training !!!")
                        errfile.write(traceback.format_exc())

    except KeyboardInterrupt:
        print("...Training finishing...")

    except Exception:
        print("!!! Exception caught on training !!!")
        print(traceback.format_exc())
        with open(run_name + ".err", "a") as errfile:
            errfile.write("!!! Exception caught on training !!!")
            errfile.write(traceback.format_exc())

    finally:
        if not finish_event.is_set():
            print("Train process set finish flag.")
            finish_event.set()

        print("Training finished.")
