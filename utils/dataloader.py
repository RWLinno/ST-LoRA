import os
import pickle
import torch
from torch import Tensor
import numpy as np
import threading
import multiprocessing as mp
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as DL

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class StandardScaler2():
    """
    Standard scaler for input normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_dataset(data_path, args):
    # 额外处理 air_tiny 数据集
    # if args.dataset == 'AirTiny':
    #     # https://github.com/yoshall/AirFormer/blob/main/src/utils/helper.py
    #     args.logger.info('Data shape: ' + str(ptr['data'].shape))
    #     dataloader = {}
    #     for cat in ['train', 'val', 'test']:
    #         idx = np.load(os.path.join(args.data_path+'/'+args.years +'/'+ category + '.npz'))
    #         dataloader['x_' + cat] = cat_data['x']
    #         dataloader['y_' + cat] = cat_data['y']

    #     scalers = []
    #     for i in range(args.output_dim):
    #         scalers.append(StandardScaler2(mean=data['x_train'][..., i].mean(),std=data['x_train'][..., i].std()))

    #     # Data format
    #     for category in ['train', 'val', 'test']:
    #         # normalize the target series (generally, one kind of series)
    #         for i in range(args.output_dim):
    #             data['x_' + category][..., i] = scalers[i].transform(data['x_' + category][..., i])
    #             data['y_' + category][..., i] = scalers[i].transform(data['y_' + category][..., i])

    #         new_x = Tensor(data['x_' + category])
    #         new_y = Tensor(data['y_' + category])
    #         processed[category] = TensorDataset(new_x, new_y)

    #     results['train_loader'] = DL(processed['train'], args.batch_size)
    #     results['val_loader'] = DL(processed['val'], args.batch_size)
    #     results['test_loader'] = DL(processed['test'], args.batch_size)

    #     print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
    #                                                 len(results['val_loader'].dataset),
    #                                                 len(results['test_loader'].dataset)))
    #     scaler = StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std())
    #     results['scaler'] = scaler
    #     return results, scaler

    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    args.logger.info('Data shape: ' + str(ptr['data'].shape))
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_length, args.horizon, args.batch_size, args.logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)

def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
        'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
        'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
        'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
        'PEMS03': [base_dir+'pems03', base_dir+'pems03/pems03_rn_adj.npy', 358],
        'PEMS04': [base_dir+'pems04', base_dir+'pems04/pems04_rn_adj.npy', 307],
        'PEMS07': [base_dir+'pems07', base_dir+'pems07/pems07_rn_adj.npy', 883],
        'PEMS08': [base_dir+'pems08', base_dir+'pems08/pems08_rn_adj.npy', 170],
        'PEMSBAY': [base_dir+'pemsbay', base_dir+'pemsbay/pemsbay_rn_adj.npy', 325],
        'METRLA': [base_dir+'metrla', base_dir+'metrla/metrla_rn_adj.npy', 207],
        'TAXIBJ': [base_dir+'TaxiBJ', base_dir+'TaxiBJ/TAXIbj_rn_adj.npy', 1024],
        'AirTiny': [base_dir+'AirTiny', base_dir+'AirTiny/airtiny_rn_adj.npy', 1085]
        }
    assert dataset in d.keys()
    return d[dataset]