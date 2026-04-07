# Benchmark of data-centric AI methods mitigating segmentation label noise in federated learning

## Motivation / Objective

## Data

### Datasets and Noise
...

### `nnUNet-FL`'s Planning and Preprocessing
The nnUNet model is a self-configuring model adjusting its architecture, training scheme and more based on the processed data.
For the FL case that distributed data of FL clients is very heterogene, it can be the case that different FL clients configure different models in the self-configuration phase.
For this case, the aggregation of model weights during the FL training would fail!

The solve this problem, we need to do not only the model training in a federated manner, but also already the self-configuration phase.
To configure the model accordingly, we have to run the scipt `nnunet_fed_preparation.py`, which extracts data fingerprints per Fl client, centrally averages them and subsequently plans the experiments and preprocesses the data accordingly.

**Example:**
Prepare envs:
```
export nnUNet_raw="/home/m391k/Documents/my_documents/Publications/fed_noisy_label_benchmark/data/MAMA-MIA/nnUNet_raw"
export nnUNet_preprocessed="/home/m391k/cluster-data-1/data_noisy-seg-label-benchi/nnunet-preprocessed-blosc_4real"
export nnUNet_results=""    # not necessary for planning and preprocessing
```

Execute planning and preprocessing:
```
python3 ./src/data/utils/nnunet_fed_preparation.py --dataset_ids "505 506 507 508 509" --configuration "3d_fullres" --planner "nnUNetPlannerResEncM" --plans_name "nnUNetResEncUNetMPlans" --verify_dataset_integrity
```

### Reqirements in virtual environment .venv:

1. Initialize the environment and install dependencies:
```make```

2. Update the environment with new packages or submodule changes:
```make update```

3. Clean the environment:
```make clean```

## Figure generation

Overall segmentation performance boxplots:
```
python3 ./src/eval/results_processing/visualize_results.py
```

Overall ranking stability plots:
```
python3 ./src/eval/results_processing/visualize_ranking.py --output-dir ./results/segmentation_results/ranking_stability
```

Overall partial noise settings (roc, roa) comparison AND roc-clean vs roc-noisy clients comparison:
```
python3 ./src/eval/results_processing/roc_clean_vs_noisy_clients_global_comparison.py --output-dir ./results/segmentation_results/partial_noise_comparison --figure paired_dot
```
