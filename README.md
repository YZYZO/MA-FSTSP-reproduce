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
Project options are configured in `config.py`. Set `RUN_FULL_EXPERIMENTS=True` there to run the paper-scale suite. Boston/Cambridge OSM download is disabled by default and can be explicitly enabled with `ALLOW_OSM_DOWNLOAD=True`.

## H2H distance backend

Road-network files are selected explicitly by `MANHATTAN_GRAPH_PATH`, `MANHATTAN_BASELINE_GRAPH_PATH`, and `BOSTON_GRAPH_PATH` in `config.py`. Missing files fail with their full path unless a separate fallback option is explicitly enabled. Legacy all-pairs JSON/pickle files are no longer loaded, rewritten, or deleted.

Build the native backend once on Windows:

```powershell
D:\Anaconda3\envs\MA-FSTSP\python.exe scripts\build_h2h_native.py --release
```

Linux servers use the same script with `--compiler g++ --release`. H2H indexes are versioned by a complete SHA-256 graph fingerprint under `datasets/indexes/`; concurrent processes share the finished read-only mmap index and only one process may build a missing cache.

`datasets/nyc.graphml` is intentionally blocked while `H2H_ENABLE_55K=False`. Enable it only on the approximately 200 GB RAM server. Small local fixtures use the 4,333-node standardized Manhattan baseline instead.

Run the explicit local phase-6 acceptance suite (temporary indexes only) with:

```powershell
$env:H2H_RUN_LOCAL_ACCEPTANCE = "1"
D:\Anaconda3\envs\MA-FSTSP\python.exe -m unittest tests.test_h2h_phase6_acceptance -v
```

On the Linux server, keep the default guard in `config.py` and grant one-process authorization through the dedicated runner:

```bash
python scripts/run_h2h_server_acceptance.py \
  --confirm-server-55k --compiler g++ \
  --worker-counts 1,4,8,16 --customer-counts 20
```

The runner requires Linux and at least 150 GiB RAM by default, recompiles the native `.so`, validates 100,000 directed queries from 200 Dijkstra sources, measures worker scaling, runs the 5-depot/20-customer/3-drone case, and atomically writes `results/h2h-server-55k-acceptance.json`. See `docs/H2H_SERVER_ACCEPTANCE.md` before running it.
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
