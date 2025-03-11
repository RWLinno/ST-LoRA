# ST-LoRA
This code is a PyTorch implementation of our paper **"Low-rank Adaptation for Spatio-Temporal Forecasting"**.

**<font color='red'>[Highlight]</font> The updated code will be released with more baselines and modules upon acceptance of the paper.**
**<font color='red'>Part of the information will be hidden during the review phase. The latest source code will be released when the paper is accepted.</font>**

## 🔗Citing  ST-LoRA
(🌟It's very important for me~~~)

If you find this resource helpful, please consider to star this repository and cite our research:

## 💿Requirements

- python >= 3.7
- torch==1.13.1

All dependencies can be installed using the following command:
```
conda create -n stlora python==3.7
conda activate stlora
pip install torch==1.13.1 torch_geometric torchinfo
pip install -r requirements.txt
```

## 📚repo structure

-  main.py
- data
- generate_training_data -> refer to 'Graph-WaveNet'
  - rawdata.h5 -> year_dataset/(his.npz, idx_test.npy, idx_train.npywe4, idx_val.npy)
- experiments -> expr. log
- save -> model / results
- src -> source code for stlora

## 📦Dataset
Our experiments utilize six public traffic datasets: PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, and PEMS-BAY.
You can download these datasets via:
- **Option 1**: [Google Drive](https://drive.google.com/drive/folders/1vtfAlMufZJxzoLsdJXFasE39pfc1Xcqn?usp=sharing)
- **Option 2**: Run `./download_datasets.sh`

##### Dataset Statistics

| Dataset | #Nodes | #Frames | Time Range |
|---------|--------|---------|------------|
| METR-LA | 207 | 34,272 | 03/2012 – 06/2012 |
| PEMS-BAY | 325 | 52,116 | 01/2017 – 06/2017 |
| PEMS03 | 358 | 26,208 | 09/2018 – 11/2018 |
| PEMS04 | 307 | 16,992 | 01/2018 – 02/2018 |
| PEMS07 | 883 | 28,224 | 05/2017 – 08/2017 |
| PEMS08 | 170 | 17,856 | 07/2016 – 08/2016 |

## ⭐Quick Start

```
python main.py [-dataset] [-device] [-pre_train] [-seed] [-epochs] ...
```

Examples for all parameters in commands. You can modify some of the default parameters in `./src/utils/args.py`  contained:

```
--seed=998244353 
--batch_size=64 
--seq_length=12 
--horizon=12 
--input_dim=3 
--output_dim=1 
--mode=train 
```

It is also recommended that you train with the following commands and modifiable parameters:

```
python main.py --device=cuda:1 --dataset=PEMS08 --years=2016 --stlora
# using python main.py to train original models
# You need to modify the backbone model in the `main.py` header file
```


### 🎯Training from scratch

##### run PEMS03/PEMS04/PEMS07/PEMS08 be like:

```
# original model
python main.py --device=cuda:1 --dataset=PEMS04 --years=2018 --mode=train
# use st-lora for adjustment
python main.py --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24 --num_mlrfs=4 

# Enhance STGCN
python main.py --device=cuda:0 --dataset=PEMS04 --years=2018 --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24

# Enhance Graph WaveNet
python main.py --device=cuda:0 --dataset=PEMS03 --years=2018 --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24

# Enhance AGCRN
python main.py --device=cuda:0 --dataset=PEMS07 --years=2017 --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24

# Enhance STID
python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24
```

### 🧪 Fine-tuning using LoRA
Stay tuned for the latest repo/experiments
Our experiments show significant improvements across multiple metrics (MAE, RMSE, MAPE) when applying ST-LoRA to backbone models. For detailed results, please refer to our paper.


### 📈 Visualization
Stay tuned for additional visualization examples in the `tutorials/` directory.


## 🙏 Acknowledgements
The acknowledgment will be hidden during the paper review phase  

