import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

base = {
    'diffusion': {
        ## model
        'model': 'models.ConditionalUnet1D',
        'diffusion': 'models.CFM',
        'horizon': 64,
        # 'global_cond_dim': 9,
        'global_cond_dim': 8,
        'n_diffusion_steps': 100,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'clip_denoised': True,
        'use_wavelet': False,
        'cont': None,
        ## serialization
        'logbase': 'logs/pose_trajectory_athit_notrajpart',
        'prefix': 'cfm/',
        'exp_name': watch(diffusion_args_to_watch),
        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 200000,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 5000,
        'sample_freq': 0,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },
}
