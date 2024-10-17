import os
import time
import argparse
import numpy as np
import torch.optim as optim
from src.__init__ import *
import src.loralib as lora

def main():
    start_time = time.time()
    args = get_config()
    args.log_dir = './experiments/{}/'.format(args.model)
    init_seed(args.seed) # Set random seed
    device = torch.device(args.device) # Set device
    args.data_path, args.adj_path, args.node_num = get_dataset_info(args.dataset)
    args.logger.info('Adj path: ' + args.adj_path)

    # Load adjacency matrix
    args.adj_mx = load_adj_from_numpy(args.adj_path)
    if args.nor_adj:
        args.adj_mx = normalize_adj_mx(args.adj_mx, args.adj_type)
    args.supports = [torch.tensor(i).to(args.device) for i in args.adj_mx]

    # Load dataset
    args.loss_fn = masked_mae
    args.dataloader, args.scaler = load_dataset(args.data_path, args)
    engine = get_engine(args)

    if args.stlora:
        engine.model = STLoRA(device=args.device,
                                node_num=args.node_num,
                                input_dim=args.input_dim,
                                output_dim=args.output_dim,
                                horizon=args.horizon,
                                model=engine.model,
                                supports=args.supports,
                                frozen=args.frozen,
                                lagcn=args.lagcn,
                                embed_dim=args.embed_dim,
                                num_layers = args.num_nalls,
                                num_blocks = args.num_lablocks,
                                la_dropout=args.last_dropout,
                                last_lr=args.last_lr,
                                last_weight_decay=args.last_weight_decay,
                                last_pool_type=args.last_pool_type
                                )
        engine.model.to(engine._device)

    if args.pre_train:
        pretrained_dict = torch.load('./save/'+args.pre_train, map_location=device)
        model_dict = engine.model.state_dict()
        model_dict.update(pretrained_dict)
        engine.model.load_state_dict(pretrained_dict, strict=False)
       
    #print(engine.model)
    # for name, value in engine.model.named_parameters():
    #     if value.requires_grad:
    #         print(name)
    
    train_time = time.time()
    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)
    
    if args.save:
        if not os.path.exists('./save'):
            os.makedirs('./save')
        torch.save(engine.model.state_dict(), './save/'+args.save)


    print_trainable_parameters(engine.model)
    print(args.model,args.mode," finished!! thank you!!")
    os.system('start wb.mp3')
    end_time = time.time()
    print("total run time: {} s".format(end_time - start_time))
    print("total train time: {} s".format(end_time - train_time))

if __name__ == "__main__":
    main()