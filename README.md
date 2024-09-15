# ST-LoRA

This code is a PyTorch implementation of our paper **"Low-rank Adaptation for Spatio-Temporal Forecasting"**. [arXiv](https://arxiv.org/abs/2404.07919)

## 🔗Citing  ST-LoRA
(🌟It's very important for me~~~)

If you find this resource helpful, please consider to star this repository and cite our research:
```
@article{ruan2024low,
  title={Low-rank Adaptation for Spatio-Temporal Forecasting},
  author={Ruan, Weilin and Chen, Wei and Dang, Xilin and Zhou, Jianxiang and Li, Weichuang and Liu, Xu and Liang, Yuxuan},
  journal={arXiv preprint arXiv:2404.07919},
  year={2024}
}
```


## 💿Requirements

---

- python >= 3.7

- torch==1.13.1

Other dependencies can be installed using the following command:

```
pip install -r requirements.txt
```

## 📚repo structure

-  main.py
- generate_training_data -> refer to 'Graph-WaveNet'
  - rawdata.h5 -> year_dataset/(his.npz, idx_test.npy, idx_train.npywe4, idx_val.npy)
- experiments -> expr. log
- save -> model / results
- src -> source code for stlora

## 📦Dataset

### Download Data

- You can download datasets used in the paper via this link: [Google Drive](https://drive.google.com/drive/folders/1vtfAlMufZJxzoLsdJXFasE39pfc1Xcqn?usp=sharing)


### Experiments

We conduct experiments on 

##### Quick Start

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
# using python main.py to train orginal models
# You need to modify the backbone model in the `main.py` header file
```

##### Comparison with MLP (using)

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



### Fine-tuning



### 📈 Visualization



## Acknowledgements
I would like to sincerely thank **Prof. Yuxuan Liang** and the [Citymind](https://citymind.top/about-us/) Group from The Hong Kong University of Science and Technology (Guangzhou) for their valuable support and guidance throughout the ST-LoRA project. Their contributions were instrumental to the success of this work.

