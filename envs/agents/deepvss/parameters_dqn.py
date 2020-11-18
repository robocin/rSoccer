import math

MODEL_HYPERPARAMS = {
    'agent': 'DQN',
    'model_type': 'dqn',  # 'dqn| dueling_dqn | noisy_dqn
    'save_model_frequency': 500000,
    'replay_size': 5000000,
    'replay_initial': 100000,
    'target_net_sync': 500000,
    'epsilon_frames': 10 ** 5,
    'epsilon_high_start': 1.0,
    'epsilon_high_final': 0.02,
    'epsilon_low': 0.02,
    'epsilon_decay': 0.995,
    'eval_freq_matches': 2,
    'eval_opponent_exp': 0.85,
    'dqn_hidden': 256,
    'dqn_dropout_p': 0,
    'learning_rate': 0.0001,
    'weight_decay': 0,
    'gamma': 0.95,
    'unroll_steps': 2,
    'batch_size': 128,
    'stop_reward': 100,
    'data_path': 'data_path'
}
