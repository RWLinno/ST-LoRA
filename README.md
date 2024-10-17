# ST-LoRA

This code is a PyTorch implementation of our paper **"Low-rank Adaptation for Spatio-Temporal Forecasting"**.

**<font color='red'>[Highlight]</font> This code is the version as of March 14, 2024, and the updated code will be released upon acceptance of the paper.**
**<font color='red'>Part of the information will be hidden during the review phase.</font>**

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

You can download datasets used in the paper via this link: [Google Drive](https://drive.google.com/drive/folders/1vtfAlMufZJxzoLsdJXFasE39pfc1Xcqn?usp=sharing)
or use `./download_datasets.sh` to download datasets.



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

##### Comparison with MLP

```
python main.py --stlora --mlp
```



### 🎯Training from scratch

##### run PEMS03/PEMS04/PEMS07/PEMS08 be like:

```
# original model
python main.py --device=cuda:1 --dataset=PEMS04 --years=2018 --mode=train
# use st-lora for adjustment
python main.py --mode=train --stlora --mlp --num_nalls=4 --embed_dim=24 --num_mlrfs=4 
```

### Fine-tuning using LoRA
Stay tuned for the latest repo/experiments


### 📈 Visualization
Stay tuned for the latest repo/tutorials


## Acknowledgements
The acknowledgment will be hidden during the paper review phase  

