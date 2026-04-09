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

### Data analysis results figures

#### Multi-rater label analysis
To visualize the variability among the raters leading to label noise, we visualize:
- class-wise Fleiss' Kappa scores among the rater's masks
- class-wise mean Dice scores of consensus masks vs each of the rater's masks (mean over raters)
- class-wise mean HD95 scores of consensus masks vs each of the rater's masks (mean over raters)
- class-wise instance-level mean F1 score of consensus masks vs each of the rater's masks (mean over raters)
- mean Class-confusion matrix of consensus masks vs each of the rater's masks (mean over raters)

Generate plot via:
```
python3 ./src/data/data_analysis/visualize_multirater_consensus_violin.py \
    --input_json ./results/consensus_analysis/<YOUR-DATASET>/multirater_consensus.json \
    --output_png ./results/consensus_analysis/<YOUR-DATASET>/fk_dice_hd95_if1_clsconf.png 
```

#### Consensus- vs. noisy label analysis
To visualize the degree and type of label noise, we compare the noisy label masks against their consensus counterpart:
- class-wise Dice scores of consensus masks vs noisy mask
- class-wise HD95 scores of consensus masks vs noisy mask
- class-wise instance-level F1 score of consensus masks vs noisy mask
- Class-confusion matrix of consensus masks vs noisy mask

Generate plot via:
```
python3 ./src/data/data_analysis/visualize_perclass_boxplots.py \
    --input_json ./results/noise_analysis/noise_analysis_results_clean<DATASET-IDS-OF-YOUR-DATASET>.json \
    --output_dir ./results/noise_analysis/<YOUR-DATASET>/ \
    --requested_row_width 27 \
    --requested_row_height 5.4 \
    --requested_row_keep_ratio
```

To visualize scatter points manifesting different label noise types, we plot:
- HD95 to capture contour differences
- instance-level F1 score to capture missing/additional labels
- class-confusion score to capture swapped labels

Generate plot via:
```
python3 ./src/data/data_analysis/visualize_hd95_f1_confusion.py --json_path ./results/noise_analysis/noise_analysis_results_clean<DATASET-IDS-OF-YOUR-DATASET>.json --output ./results/noise_analysis/<YOUR-DATASET>/hd95_vs_f1_vs_confusion.png --level instance --figsize 11 9
```

### Segmentation results figures

By default, the result-processing scripts that load `nnUNet` evaluation outputs for plots combine checkpoints from both of these roots:

```
/home/m391k/cluster-data/checkpoints/nnUNet_results
/home/m391k/juwels/checkpoints/nnUNet_results
```

This currently applies to:

- `src/eval/results_processing/visualize_results.py`
- `src/eval/results_processing/visualize_ranking.py`
- `src/eval/results_processing/partial_noisy_scenarios_global_comparison.py`
- `src/eval/results_processing/roc_clean_vs_noisy_clients_global_comparison.py`

The scripts merge experiment paths from both locations and deduplicate them before filtering to the configured folds.
If you set the environment variable `nnUNet_results` or pass `--nnunet-results-root` to scripts that support it, that single path is used instead of the two default roots.

Overall segmentation performance boxplots:
```
python3 ./src/eval/results_processing/visualize_results.py
```

Overall ranking stability plots:
```
python3 ./src/eval/results_processing/visualize_ranking.py --output-dir ./results/segmentation_results/ranking_stability
```

Overall partial noise settings (`roa` vs `roc`) comparison:
```
python3 ./src/eval/results_processing/partial_noisy_scenarios_global_comparison.py --output-dir ./results/segmentation_results/partial_noise_comparison --figure paired_dot
```

Per-client -- roc-clean vs roc-noisy clients comparison:
```
python3 ./src/eval/results_processing/roc_clean_vs_noisy_clients_global_comparison.py --output-dir ./results/segmentation_results/partial_noise_comparison --figure paired_dot
```
