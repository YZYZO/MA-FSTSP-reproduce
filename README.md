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
`run_full_experiments()` runs the 1K, 11K, and 55K NYC road networks in that order. Each map initializes its pairwise road/drone distances once, then reuses them for 50, 100, and 150 customers before releasing the distance matrices. The 1K scale uses 5 depots and 3 drones per truck; the 11K and 55K scales use 10 depots and 4 drones per truck. The 11K reproduction keeps the historical `boston_11k` result name, while the 55K experiment uses `datasets/nyc.graphml` and the `manhattan_55k` result name.

The default command runs 100 instances for every map/customer-size combination and is intended for the large-memory server. Run the lightweight orchestration tests locally without loading the real maps:

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -m unittest tests.test_map_scale_batch_initialization -v
```

Boston/Cambridge OSM download settings are colocated with the map-loading implementation in `problem.py`. Download and refresh are disabled by default, and both switches must be enabled before the code accesses Overpass.
To generate all the figures in the paper, you can use 
```bash
python plot.py
```

To visualize the latest saved 1K/11K solution batch without re-running the optimizer:

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -c "import plot; plot.plot_large_road_experiment_results(customer_count=100)"
```

New road-network NPZ files include the sampled depots/customers, final truck/drone routes, and compact phase telemetry. The plotting code selects the median-cost instance by default and writes an interactive HTML map plus a JSON summary.

Outputs are written under `results/`:

- `results/manhattan/data`, `results/manhattan/figures`, `results/manhattan/maps`
- `results/boston/data`, `results/boston/figures`, `results/boston/maps`
- `results/small/data`, `results/small/figures`

## Citation
If you find our research helpful for your work, please consider starring this repo and citing our paper.
