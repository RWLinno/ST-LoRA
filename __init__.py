import random
import torch
from .base.engine import *
from .model import *
from .models.__init__ import *
from .engines.__init__ import *
from .utils.__init__ import *
from fastdtw import fastdtw
import os

my_name = 'rwlinno'
its_name = 'GWN-LoRA'

# 手动设置随机种子
def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 打印可学习参数率
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _,param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def get_engine(args,**kwargs):
    if args.model == 'agcrn':
        model = AGCRN(node_num=args.node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                embed_dim=args.embed_dim,
                rnn_unit=args.rnn_unit,
                num_layer=args.num_layer,
                cheb_k=args.cheb_k,
                horizon=args.horizon
                )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None
        engine = AGCRN_Engine(device=args.device,
                    model=model,
                    dataloader=args.dataloader,
                    scaler=args.scaler,
                    sampler=None,
                    loss_fn=args.loss_fn,
                    lrate=args.lrate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    clip_grad_value=args.clip_grad_value,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    log_dir=args.log_dir,
                    logger=args.logger,
                    seed=args.seed
                    )

    elif args.model == 'gwnet':
        args.norm_adj  = normalize_adj_mx(args.adj_mx, args.adj_type)
        supports = [torch.tensor(i).to(args.device) for i in args.norm_adj]
        model = GWNET(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    supports=supports,
                    adp_adj=args.adp_adj,
                    dropout=args.dropout,
                    residual_channels=args.init_dim,
                    dilation_channels=args.init_dim,
                    skip_channels=args.skip_dim,
                    end_channels=args.end_dim,
                    horizon=args.horizon
                    )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None
        engine = BaseEngine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=args.loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )
    elif args.model == 'lstm':
        model = LSTM(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    init_dim=args.init_dim,
                    hid_dim=args.hid_dim,
                    end_dim=args.end_dim,
                    layer=args.layer,
                    dropout=args.dropout,
                    horizon=args.horizon
                    )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None
        engine = BaseEngine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=args.loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )

    elif args.model == 'stgcn':
        args.adj_mx = args.adj_mx - np.eye(args.node_num)
        gso = normalize_adj_mx(args.adj_mx, 'scalap')[0]
        gso = torch.tensor(gso).to(args.device)
        Ko = args.seq_length - (args.Kt - 1) * 2 * args.block_num
        blocks = []
        blocks.append([args.input_dim])
        for l in range(args.block_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([args.horizon])
        model = STGCN(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    gso=gso,
                    blocks=blocks,
                    Kt=args.Kt,
                    Ks=args.Ks,
                    dropout=args.dropout,
                    horizon=args.horizon
                    )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        engine = BaseEngine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=args.loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )
        
    elif args.model == 'd2stgnn':
        adj_mx = normalize_adj_mx(args.adj_mx, 'doubletransition')
        args.adjs = [torch.tensor(i).to(args.device) for i in adj_mx]
        cl_step = args.cl_epoch * args.dataloader['train_loader'].num_batch
        warm_step = args.warm_epoch * args.dataloader['train_loader'].num_batch

        model = D2STGNN(node_num=args.node_num,
                        input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        model_args=vars(args)
                        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5)
        engine = D2STGNN_Engine(device=args.device,
                                model=model,
                                dataloader=args.dataloader,
                                scaler=args.scaler,
                                sampler=None,
                                loss_fn=args.loss_fn,
                                lrate=args.lrate,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                clip_grad_value=args.clip_grad_value,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                log_dir=args.log_dir,
                                logger=args.logger,
                                seed=args.seed,
                                cl_step=cl_step,
                                warm_step=warm_step,
                                horizon=args.horizon
                                )
    elif args.model == 'dcrnn':
        model = DCRNN(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    device=args.device,
                    adj_mx=args.adj_mx,
                    n_filters=args.n_filters,
                    max_diffusion_step=args.max_diffusion_step,
                    filter_type=args.filter_type,
                    num_rnn_layers=args.num_rnn_layers,
                    cl_decay_steps=args.cl_decay_steps
                    )

        loss_fn = masked_mae #
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        steps = [10, 50, 90]  # CA: [5, 50, 90], others: [10, 50, 90]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1, verbose=True)

        engine = DCRNN_Engine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )
        
    elif args.model == 'astgcn':
        adj_mx = args.adj_mx - np.eye(args.node_num)
    
        adj = np.zeros((args.node_num, args.node_num), dtype=np.float32)
        for n in range(args.node_num):
            idx = np.nonzero(adj_mx[n])[0]
            adj[n, idx] = 1

        L_tilde = normalize_adj_mx(adj, 'scalap')[0]
        cheb_poly = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in calculate_cheb_poly(L_tilde, args.order)] 

        model = ASTGCN(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    device=args.device,
                    cheb_poly=cheb_poly,
                    order=args.order,
                    nb_block=args.nb_block,
                    nb_chev_filter=args.nb_chev_filter,
                    nb_time_filter=args.nb_time_filter,
                    time_stride=args.time_stride
                    )
        
        loss_fn = masked_mae #
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

        engine = ASTGCN_Engine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )
    
    elif args.model == 'stgode':
        args.adj_mx = args.adj_mx - np.eye(args.node_num)
        args.sp_matrix = args.adj_mx + np.transpose(args.adj_mx)
        args.sp_matrix = normalize_tensor(args.sp_matrix).to(args.device)
        args.se_matrix = construct_se_matrix(args.data_path, args)
        args.se_matrix = normalize_tensor(args.se_matrix).to(args.device)

        model = STGODE(node_num=args.node_num,
                        input_dim=args.input_dim,
                        output_dim=args.output_dim,
                        A_sp=args.sp_matrix,
                        A_se=args.se_matrix
                        )

        loss_fn = masked_mae
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        engine = BaseEngine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )
    elif args.model == 'dgcrn':
        
        model = DGCRN(node_num=args.node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    device=args.device,
                    predefined_adj=supports,
                    gcn_depth=args.gcn_depth,
                    rnn_size=args.rnn_size,
                    hyperGNN_dim=args.hyperGNN_dim,
                    node_dim=args.node_dim,
                    middle_dim=2,
                    list_weight=[0.05, 0.95, 0.95],
                    tpd=args.tpd,
                    tanhalpha=args.tanhalpha,
                    cl_decay_step=args.cl_decay_step,
                    dropout=args.dropout
                    )
        
        loss_fn = masked_mae
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None

        engine = DGCRN_Engine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed,
                            step_size=args.step_size,
                            horizon=args.horizon
                            )
    elif args.model == 'dstagnn':
        args.adj_mx = args.adj_mx - np.eye(args.node_num)
        adj = np.zeros((args.node_num, args.node_num), dtype=np.float32)
        for n in range(args.node_num):
            idx = np.nonzero(args.adj_mx[n])[0]
            adj[n, idx] = 1
        L_tilde = normalize_adj_mx(adj, 'scalap')[0]

        cheb_poly = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in calculate_cheb_poly(L_tilde, args.order)]
        adj = torch.tensor(adj).to(args.device)

        model = DSTAGNN(node_num=args.node_num,
                input_dim=args.input_dim,
                output_dim=args.output_dim,
                device=args.device,
                cheb_poly=cheb_poly,
                order=args.order,
                nb_block=args.nb_block,
                nb_chev_filter=args.nb_chev_filter,
                nb_time_filter=args.nb_time_filter,
                time_stride=args.time_stride,
                adj_pa=adj,
                d_model=args.d_model,
                d_k=args.d_k,
                d_v=args.d_k,
                n_head=args.n_head
                )
        
        loss_fn = masked_mae
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None

        engine = DSTAGNN_Engine(device=args.device,
                                model=model,
                                dataloader=args.dataloader,
                                scaler=args.scaler,
                                sampler=None,
                                loss_fn=loss_fn,
                                lrate=args.lrate,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                clip_grad_value=args.clip_grad_value,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                log_dir=args.log_dir,
                                logger=args.logger,
                                seed=args.seed
                                )
    elif args.model == 'stae':
        model =  STAEformer(num_nodes=args.node_num,
                            input_dim=args.input_dim,
                            output_dim=args.output_dim,
                            device=args.device,
                            horizon=args.horizon,
                            in_steps=12,
                            out_steps=12,
                            steps_per_day=288,
                            input_embedding_dim=24,
                            tod_embedding_dim=24,
                            dow_embedding_dim=24,
                            spatial_embedding_dim=0,
                            adaptive_embedding_dim=80,
                            feed_forward_dim=256,
                            num_heads=4,
                            num_layers=3,
                            dropout=0.1,
                            use_mixed_proj=True,
                )
        loss_fn = masked_mae
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        scheduler = None

        engine = BaseEngine(device=args.device,
                            model=model,
                            dataloader=args.dataloader,
                            scaler=args.scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.lrate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=args.log_dir,
                            logger=args.logger,
                            seed=args.seed
                            )

    return engine

def construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    data = ptr['data'][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
    data_mean = data_mean.T
    
    dist_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i, node_num):
            dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in range(node_num):
        for j in range(i):
            dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix

def normalize_tensor(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))