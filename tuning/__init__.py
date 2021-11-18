from ray import tune

def load_configs(exp_name):
    # Ray Option
    if exp_name == 'tune_1':
        config = {
                    'train/lr_cls': tune.loguniform(1e-4, 1e-1),
                    'train/lr_attn': tune.loguniform(1e-4, 1e-1),
                    'train/attn_loss': tune.choice(['CE', 'CE+Superior'])
                 }
    else:
        raise('Select Proper Exp-Name')

    return config
