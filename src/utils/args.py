import argparse
import os
import sys
import logging
from src.config import GWNConfig as cf

def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)
    
    return logger

def get_config():
    parser = get_public_config()
    args = parser.parse_args()
    addition = get_model_args(args.model)
    for key, value in addition.items():
        setattr(args, key, value)
    log_dir = './experiments/{}/{}/'.format(args.model, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    args.logger = logger
    args.log_dir = log_dir
    return args

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=cf.device)
    parser.add_argument('--dataset', type=str, default=cf.dataset)
    parser.add_argument('--years', type=str, default=cf.years)
    parser.add_argument('--model', type=str, default=cf.model)
    parser.add_argument('--seed', type=int, default=cf.seed)
    parser.add_argument('--batch_size', type=int, default=cf.batch_size)
    parser.add_argument('--seq_length', type=int, default=cf.seq_length)
    parser.add_argument('--horizon', type=int, default=cf.horizon)
    parser.add_argument('--input_dim', type=int, default=cf.input_dim)
    parser.add_argument('--output_dim', type=int, default=cf.output_dim)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=int, default=5)
    parser.add_argument('--adj_type', type=str, default='doubletransition')

    parser.add_argument('--stlora', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--nor_adj', action='store_true')
    parser.add_argument('--lagcn', action='store_true')
    parser.add_argument('--pre_train', type=str, default="")
    parser.add_argument('--save', type=str, default="test.pth")
    parser.add_argument('--embed_dim', type=int, default=12)
    parser.add_argument('--num_nalls', type=int, default=4)
    parser.add_argument('--num_lablocks', type=int, default=1)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--last_dropout', type=float, default=0.3)
    parser.add_argument('--last_lr', type=float, default=1e-3)
    parser.add_argument('--last_weight_decay', type=float, default=1e-4)
    parser.add_argument('--last_pool_type', type=str, default="mean")

    return parser

def get_model_args(model_name):
    model_args = {}
    if model_name == 'gwnet':
        model_args = {
                      'adp_adj' : 1,
                      'init_dim' : 32,
                      'skip_dim' : 256,
                      'end_dim' : 512,
                      }
    elif model_name == 'stgcn':
        model_args = {'Kt' : 3,
                    'Ks' : 3,
                    'block_num' : 2,
                    'step_size' : 10,
                    'end_dim' : 512,
                    'gamma' : 0.95,
                    }
    elif model_name == 'agcrn':
        model_args = {
            'rnn_unit' : 64,
            'num_layer' : 2,
            'cheb_k' : 2,      
        }
    elif model_name == 'lstm':
        model_args = {
            'init_dim':32,
            'hid_dim':64,
            'end_dim':512,
            'layer':2
        }
    elif model_name == 'd2stgnn':
        model_args = {
            'num_feat':1,
            'num_hidden':32,
            'node_hidden':12,
            'time_emb_dim':12,
            'layer':5,
            'k_t' : 3,
            'k_s' : 2,
            'gap' : 3,
            'cl_epoch':3,
            'warm_epoch':30,
            'tpd':96
        }
    elif model_name == 'dcrnn':
        model_args = {
            'n_filters':64,
            'max_diffusion_step':2,
            'filter_type':'doubletransition',
            'num_rnn_layers':2,
            'cl_decay_steps':2000
        }
    elif model_name == 'astgcn':
        model_args = {
            'order':3,
            'nb_block':2,
            'nb_chev_filter':64,
            'nb_time_filter':64,
            'time_stride':1
        }
    elif  model_name == 'stgode':
        model_args = {
            'tpd' : 96,
            'sigma': 0.1,
            'thres': 0.6
        }
    elif model_name == 'dgcrn':
        model_args = {
            'gcn_depth': 2,
            'rnn_size': 64,
            'hyperGNN_dim': 16,
            'node_dim': 40,
            'tanhalpha': 3,
            'cl_decay_step': 2000,
            'step_size': 2500,
            'tpd': 96
        }
    elif model_name == 'dstagnn':
        model_args = {
            'order': 2,
            'nb_block': 2,
            'nb_chev_filter': 32,
            'nb_time_filter': 32,
            'time_stride': 1,
            'd_model': 512,
            'd_k': 32,
            'n_head': 3
        }
    return model_args
