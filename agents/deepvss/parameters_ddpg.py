import math


MODEL_HYPERPARAMS = {  # DDPG
    'agent': 'DDPG',
    'act_type': 'linear',
    'crt_type': 'linear',
    'save_model_frequency': 500000,
    'replay_size': 500000,
    'replay_initial': 100000,
    'target_net_sync': 1 - 1e-3,
    'epsilon_frames': 10 ** 5,
    'epsilon_high_start': 1.0,
    'epsilon_high_final': 0.4,
    'epsilon_low': 0.01,
    'epsilon_decay': 0.995,
    'eval_freq_matches': 50,
    'eval_opponent_exp': 0.99,
    'act_hidden': 256,
    'crt_hidden': 256,
    'learning_rate': 0.0001,
    'weight_decay': 1e-7,
    'gamma': 0.95,
    'unroll_steps': 2,
    'batch_size': 128,
    'stop_reward': 100,
    'ou_teta': 0.15,
    'ou_sigma': 0.2,
    'data_path': 'data_path',
    # Prioritized Experience Replay
    'per': False,
    'per_alpha': 0.6,
    'per_beta': 0.4,
}
