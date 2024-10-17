###################################
# 默认参数
###################################   
class DefaultConfig(object):
    device = 'cpu'
    dataset = 'METRLA'
    years = '2012'
    model = 'stgcn'
    seed = 998244353
    batch_size = 64
    seq_length = 12
    horizon = 12
    input_dim = 2
    output_dim = 1

class GWNConfig(object):
    device = 'cpu'
    dataset = 'PEMSBAY'
    years = '2017'
    model = 'gwnet'
    seed = 998244353
    batch_size = 64
    seq_length = 12
    horizon = 12
    input_dim = 2
    hidden_dim = 8
    output_dim = 1  
    frozen = True
    pre_train = None
