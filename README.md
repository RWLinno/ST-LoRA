# ST-LoRA
This code is a PyTorch implementation of our paper **"Low-rank Adaptation for Spatio-Temporal Forecasting"**.
🎉 Update (May 2025): This paper has been accepted by <ECML-PKDD2025>! You can check it out [here](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_120.pdf). 🔥🔥🔥

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

Examples for all parameters in commands. You can modify some of the default parameters in `./src/utils/args.py`:

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

python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --model=gwnet --mode=train
# using python main.py to train original models

python main.py --device=cuda:1 --dataset=PEMS08 --years=2016 --stlora
# You need to modify the backbone model in the `main.py` header file
```


### 🎯Training from scratch

##### run PEMS03/PEMS04/PEMS07/PEMS08 be like:

```
# Train STGNN baselines from scratch (choose one backbone with --model)
# Supported: gwnet, stgcn, agcrn, dcrnn, astgcn, d2stgnn, dstagnn, stae, mlp
python main.py --device=cuda:0 --dataset=PEMS04 --years=2018 --model=gwnet --mode=train
python main.py --device=cuda:0 --dataset=PEMS03 --years=2018 --model=stgcn --mode=train
python main.py --device=cuda:0 --dataset=PEMS07 --years=2017 --model=agcrn --mode=train
python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --model=dcrnn --mode=train

# A fast non-graph baseline
python main.py --device=cuda:0 --dataset=METRLA --years=2012 --mode=train --mlp
```

### 🧪 Fine-tuning and Enhancement with ST-LoRA 

- ST-LoRA wraps any backbone and adds node-adaptive low-rank predictors (few extra params, node-specific adjustment). For details, see our paper.
- You can optionally inject LoRA directly into backbone layers before wrapping with ST-LoRA.

Basic usage (wrap backbone with ST-LoRA):
```
python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --model=gwnet --mode=train \
  --stlora --num_nalls=4 --embed_dim=24
```

Freeze backbone and only train adapters (parameter-efficient fine-tuning):
```
python main.py --device=cuda:0 --dataset=METRLA --years=2012 --model=dcrnn --mode=train \
  --stlora --frozen --num_nalls=4 --embed_dim=24
```

Inject LoRA into backbone modules (Linear/Conv) with include/exclude filters:
```
python main.py --device=cuda:0 --dataset=PEMS04 --years=2018 --model=stgcn --mode=train \
  --backbone_lora --lora_r=8 --lora_alpha=16 --lora_dropout=0.1 \
  --lora_include=proj,ffn --lora_exclude=bn,layernorm
```

Combine both (backbone STGNNs + ST-LoRA):
```
python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --model=gwnet --mode=train \
  --backbone_lora --stlora --num_nalls=4 --embed_dim=24

# please Uncomment and give model_name at the same time
python main.py --device=cuda:0 --dataset=PEMS08 --years=2016 --mode=train --stlora --num_nalls=4 --embed_dim=24 --pre_train --load_pretrain_path='best.pt' --model_name=AGCRN
```

Optional ST-LoRA flags: `--linear` (use linear adapters inside node predictor), `--num_lablocks` (stacked predictors), `--last_pool_type` (mean/min/max/absmin/weighted).


### 📈 Visualization
Stay tuned for additional visualization examples in the `tutorials/` directory.


## 🔧 What the Modules Do and Why They Are Efficient

- ST-LoRA Wrapper (`src/model.py::STLoRA`):
  - Adds lightweight, node-adaptive low-rank predictors on top of any backbone output; supports residual fusion via multiple blocks and pooling strategies.
  - Efficient: typically ~1% extra parameters while improving performance (e.g., ~7% in our study) by explicitly modeling node-level heterogeneity.
- Node-Specific Predictor and NALL:
  - Uses low-rank linear adapters (NALL) or standard linear layers (`--linear`) across horizons, plus dropout/BN/LeakyReLU for stability.
  - Works as a small per-node function approximator over backbone outputs, enabling personalized adjustments.
- General LoRA Injection (`src/loralib/inject.py`):
  - One-line injection of LoRA into Linear/Conv layers across any backbone, with include/exclude filters for precise control.
  - Efficient: freeze backbone weights, train only LoRA factors; reduces memory and compute while keeping strong performance.

These modules are complementary: LoRA brings parameter-efficient tuning inside layers; ST-LoRA adds node-level personalization on outputs. Combined, they deliver high gains with low overhead.

## 🔗Citing  ST-LoRA
If you find this resource helpful, please consider to star this repository and cite our research:
```
@article{ruan2024low,
  title={Low-rank adaptation for spatio-temporal forecasting},
  author={Ruan, Weilin and Chen, Wei and Dang, Xilin and Zhou, Jianxiang and Li, Weichuang and Liu, Xu and Liang, Yuxuan},
  journal={arXiv preprint arXiv:2404.07919},
  year={2024}
}
```

