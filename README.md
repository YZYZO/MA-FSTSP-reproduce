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
conda create -n MA-FSTSP python=3.10
conda activate MA-FSTSP
pip install -r requirements.txt
```

## Reproduce Guide
You can find our algorithm implemented in file `src/fstsp.py` and all other baselines in the same folder. 
Run the paired Phase-1 pilot with:
```bash
python experiments.py
```
The no-argument command uses the V2 `PILOT_PROTOCOL`: Manhattan 1K, 50 customers, 10 paired instances, and a 600-second instance limit. Every partition method receives the same sampled depots and customers.

The formal protocol must be selected explicitly because it runs 100 paired instances for every map/customer-size setting:

```bash
python experiments.py --protocol formal
```

The 11K graph in this repository is labeled `NYC 11K proxy`; it is not presented as the paper's Boston road network. Run the lightweight tests locally without loading the real maps:

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" -m unittest discover -s tests -p "test_*.py"
```

The main comparison is `smst_original`, `snn`, `set_gtds_no_budget`, and `directed_set_gtds`. Main GTDS variants use all available depots and the paper's `1/speed` drone-cost factor. Epsilon, free-depot, and legacy `sqrt(2)/speed` studies are separate CLI protocols: `epsilon`, `active-depot`, and `cost-factor`.

Analyze a V2 summary and draw the required mechanism plots with:

```powershell
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" analyze_paired_results.py results\paired\phase1_pilot_v2\manhattan_1k\50\paired_summary.npz --cutoff 600
& "D:\Anaconda3\envs\MA-FSTSP\python.exe" plot_paired_mechanisms.py results\paired\phase1_pilot_v2\manhattan_1k\50\paired_summary.npz
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
- `results/nyc_proxy/data`
- `results/boston/data`, `results/boston/figures`, `results/boston/maps`
- `results/small/data`, `results/small/figures`

## Citation
If you find our research helpful for your work, please consider starring this repo and citing our paper.
