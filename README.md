# Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks

[**Ruixiao Yang**](https://scholar.google.com/citations?user=c0W8nfwAAAAJ), [**Chuchu Fan**](https://aeroastro.mit.edu/people/chuchu-fan/)

This repository provides the official implementation of our paper, "Optimization of Multi-Agent Flying Sidekick Traveling Salesman Problem over Road Networks
"[[PDF](https://arxiv.org/pdf/2408.11187)]

## Installation
Clone the repository:
```bash
git clone https://github.com/Brelliothe/MixTSP.git
```
Run the following command to install the required packages:
```bash
conda create -n MixTSP python=3.7
conda activate MixTSP
pip install -r requirements.txt
```

## Reproduce Guide
You can find our algorithm implemented in file `src/fstsp.py` and all other baselines in the same folder. 
To run all the experiments in the paper, you can use 
```bash
python experiments.py
```
Project options are configured in `config.py`. Set `RUN_FULL_EXPERIMENTS=True` there to run the paper-scale suite. Boston/Cambridge OSM download is enabled by `ALLOW_OSM_DOWNLOAD=True` and uses the radius/node limits in the same file.
To generate all the figures in the paper, you can use 
```bash
python plot.py
```

Outputs are written under `results/`:

- `results/manhattan/data`, `results/manhattan/figures`, `results/manhattan/maps`
- `results/boston/data`, `results/boston/figures`, `results/boston/maps`
- `results/small/data`, `results/small/figures`

## Citation
If you find our research helpful for your work, please consider starring this repo and citing our paper.
