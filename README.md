# Benchmark of data-centric AI methods mitigating segmentation label noise in federated learning

## Abstract

**Objective:**
Federated learning (FL) enables collaborative model training without centralizing sensitive data, making it particularly relevant for medical imaging. Yet, its deployment in medical image segmentation is challenged by real-world data imperfections across institutions, including label noise manifested as contour disagreement, missing or additional structures, or confused labels. Although federated noisy label learning (FNLL) aims to mitigate these effects, existing studies commonly evaluate methods on few datasets, simplified settings, and synthetic noise types.
We address the lack of standardized benchmarking resources for FNLL in cross-silo medical image segmentation by introducing a benchmark suite combining diverse real-world noisy datasets, deployment-relevant client-noise scenarios, and label-noise-targeted evaluation.

**Materials & Methods:**
The suite combines the curation of diverse, real-world noisy medical image segmentation datasets with a comprehensive federated segmentation framework including various client-noise scenarios and noise-targeted evaluation.
To demonstrate its capabilities, we compare representative FNLL methods across approaches, including noise-aware aggregation, robust personalization, label correction, and sample selection.

**Results:**
In-depth data analysis shows that real-world segmentation label noise occurs both in isolation and in combinations of characterized noise types.
The benchmark identifies *FedSelect* as the strongest overall FNLL method, underlines *FedAvg* as a competitive baseline, and provides an actionable decision guide to support selection of suitable FNLL strategies based on label-noise type and client-noise scenario.

**Discussion & Conclusion:**
The presented suite provides a realistic and discriminative basis for FNLL evaluation in medical image segmentation and establishes a reusable foundation for fair benchmarking, dataset-specific label-noise characterization, and future method development under realistic federated settings. Code is available at [https://github.com](https://github.com).

![](./docs/assets/FNLL_benchmarksuite_figure1.png)
*Figure 1: Segmentation label noise of various forms degrades model training and poses a particular challenge in FL, where noisy annotations are distributed across clients and cannot be centrally inspected. While FNLL methods aim to address this problem, existing literature is often limited to few and synthetic noise types, restricted client-noise scenarios, and narrow data scope. Our benchmark suite closes this gap by combining diverse real-world noisy segmentation datasets, a federated benchmarking framework, and comprehensive noise-targeted evaluation, thereby enabling FNLL method selection, dataset characterization, benchmarking on new data, and evaluation of newly developed FNLL methods.*

### Citation
```
some citation
```

## Usage

### Setup

Create and populate the Python environment from the repository requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository also provides Make targets for the local development environment:

```bash
make
make update
make clean
```

Set the nnU-Net paths before preparing data or running experiments:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Data

#### Download

Download the raw source datasets from their original providers and keep them
outside the repository. The benchmark expects the source data to be converted
into nnU-Net-style `DatasetXXX_<Name>` folders before training.

Dataset-specific preparation entry points currently live in:

```text
src/data/gleason2019/prepare.py
src/data/gleasonxai/prepare.py
src/data/mama-mia/prepare.py
src/data/mmis/prepare.py
src/data/mouse-tumor/prepare.py
src/data/riga/prepare.py
```

#### Preparation

Prepare each dataset into nnU-Net raw folders for the consensus or clean labels
and the noisy-label variants. The target layout should follow nnU-Net
conventions:

```text
nnUNet_raw/
  DatasetXXX_<DatasetNameClientA>/
    imagesTr/
    labelsTr/
    dataset.json
  DatasetYYY_<DatasetNameClientB>/
    imagesTr/
    labelsTr/
    dataset.json
```

For partially noisy scenarios, keep clean validation labels and noisy training
labels in explicit folders so they can be selected later with:

```bash
--clean_validation_dataset "<clean-validation-folder-per-client>"
--noisy_train_folder "<noisy-train-folder-per-client>"
```

#### Federated adaption of nnU-Net plan and preprocess

The nnU-Net model is self-configuring and adapts architecture, patch size, and
training settings to the processed data. In FL, independently planned clients
can produce incompatible model architectures, which breaks weight aggregation.

For this reason, planning and preprocessing must also be adapted to the
federated setting. The script `src/data/utils/nnunet_fed_preparation.py`
extracts fingerprints per client, averages them centrally, plans one compatible
experiment configuration, and preprocesses all participating clients with this
shared plan.

```bash
python3 ./src/data/utils/nnunet_fed_preparation.py \
    --dataset_ids "505 506 507 508 509" \
    --configuration "3d_fullres" \
    --planner "nnUNetPlannerResEncM" \
    --plans_name "nnUNetResEncUNetMPlans" \
    --verify_dataset_integrity
```

#### In-depth data analysis

##### Multi-rater vs. consensus mask analysis

First compute the multi-rater consensus analysis, then visualize the resulting
agreement and error metrics:

```bash
python3 ./src/data/data_analysis/analyze_multirater_consensus.py \
    --dataset_ids "500 501 502 503 504" \
    --multirater_dir /path/to/multirater_masks \
    --consensus_dir /path/to/consensus_masks
```

```bash
python3 ./src/data/data_analysis/visualize_multirater_consensus_violin.py \
    --input_json ./results/consensus_analysis/<YOUR-DATASET>/multirater_consensus.json \
    --output_png ./results/consensus_analysis/<YOUR-DATASET>/fk_dice_hd95_if1_clsconf.png
```

The visualization covers class-wise Fleiss' kappa, Dice, HD95, instance-level
F1, and class-confusion statistics against the consensus mask.

##### Noisy masks vs. consensus/clean mask analysis

Compare noisy masks against the consensus or clean reference masks:

```bash
python3 ./src/data/data_analysis/analyze_noise_clean_noisy.py \
    --clean_dataset_ids "500 501 502 503 504" \
    --noisy_dataset_ids "505 506 507 508 509" \
    --output_dir ./results/noise_analysis
```

Generate per-class boxplots:

```bash
python3 ./src/data/data_analysis/visualize_perclass_boxplots.py \
    --input_json ./results/noise_analysis/noise_analysis_results_clean<DATASET-IDS-OF-YOUR-DATASET>.json \
    --output_dir ./results/noise_analysis/<YOUR-DATASET>/ \
    --requested_row_width 27 \
    --requested_row_height 5.4 \
    --requested_row_keep_ratio
```

Generate scatter plots for contour differences, missing/additional labels, and
swapped labels:

```bash
python3 ./src/data/data_analysis/visualize_hd95_f1_confusion.py \
    --json_path ./results/noise_analysis/noise_analysis_results_clean<DATASET-IDS-OF-YOUR-DATASET>.json \
    --output ./results/noise_analysis/<YOUR-DATASET>/hd95_vs_f1_vs_confusion.png \
    --level instance \
    --figsize 12.4 10.8
```

### Run benchmarking

Run FL experiments with `src/fed/main.py`. The core switches are the dataset
IDs, client count, FL rounds, local epochs, trainer, and FNLL method.

FedAvg baseline on a three-client RIGA setup:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method fedavg \
    --dataset_ids "300 301 302" \
    --num_clients 3 \
    --num_rounds 100 \
    --num_local_epochs 5 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_FedAvg
```

IOP-FL personalization on a five-client MouseTumor setup:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method iopfl \
    --dataset_ids "505 506 507 508 509" \
    --num_clients 5 \
    --num_rounds 100 \
    --num_local_epochs 5 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_IOPFL \
    --iopfl_alpha 0.9
```

FedSelect sample/client selection on a four-client MMIA setup:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method fedselect \
    --dataset_ids "600 601 602 603" \
    --num_clients 4 \
    --num_rounds 100 \
    --num_local_epochs 5 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_FedSelect \
    --fedselect_warmup_rounds_frac 0.1 \
    --fedselect_client_select_ratio 0.4 \
    --fedselect_sample_select_ratio 0.6
```

For partially noisy client scenarios, add the clean validation and noisy train
folders:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method fedavg \
    --dataset_ids "600 601 602 603" \
    --num_clients 4 \
    --num_rounds 100 \
    --num_local_epochs 5 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_FedAvg \
    --clean_validation_dataset "<clean-val-client0> <clean-val-client1> <clean-val-client2> <clean-val-client3>" \
    --noisy_train_folder "<noisy-train-client0> <noisy-train-client1> <noisy-train-client2> <noisy-train-client3>" \
    --noise_ratio 0.5
```

### Evaluation and compilation of FNLL decisions

By default, result-processing scripts combine checkpoints from:

```text
/home/m391k/cluster-data/checkpoints/nnUNet_results
/home/m391k/juwels/checkpoints/nnUNet_results
```

Set `nnUNet_results` or pass `--nnunet-results-root` where supported to use a
single results root.

#### Run evaluation

Run nnU-Net evaluation for a single experiment ID:

```bash
python3 ./src/eval/results_processing/bootstrap_nnunet_eval.py \
    --exp_id <EXPERIMENT_ID> \
    --num-workers 8
```

Run evaluation for all registered experiments from the benchmark sheet:

```bash
python3 ./src/eval/results_processing/bootstrap_parent.py \
    --folds 0 1 2 \
    --num-workers 8
```

#### Run bootstrapping

The bootstrap evaluation files are generated by `bootstrap_nnunet_eval.py` and
`bootstrap_parent.py`. To recompute all bootstrap metrics, use:

```bash
python3 ./src/eval/results_processing/bootstrap_parent.py \
    --folds 0 1 2 \
    --force \
    --num-workers 8
```

To recompute only selected metrics:

```bash
python3 ./src/eval/results_processing/bootstrap_parent.py \
    --folds 0 1 2 \
    --force-metrics HD95 Dice \
    --num-workers 8
```

##### Result boxplot and table generation

Generate bootstrap-based result boxplots and LaTeX tables:

```bash
python3 ./src/eval/results_processing/visualize_results.py \
    --metric Dice
```

Optional dataset subset:

```bash
python3 ./src/eval/results_processing/visualize_results.py \
    --metric Dice \
    --datasets LIDC RIGA Gleason MouseTumor MMIA MMIS
```

#### Run ranking for ranking stability plot generation

Build the ranking table:

```bash
python3 ./src/eval/results_processing/ranking.py \
    --output-csv ./results/segmentation_results/bootstrap_method_rankings.csv
```

Generate ranking stability plots:

```bash
python3 ./src/eval/results_processing/visualize_ranking.py \
    --output-dir ./results/segmentation_results/ranking_stability \
    --metric Dice
```

#### Run statistical tests against FedAvg

Run paired Wilcoxon signed-rank tests comparing each FNLL method against FedAvg,
with Holm-Bonferroni correction across the four method comparisons within each
reported group:

```bash
python3 ./src/eval/results_processing/statistical_tests.py \
    --metrics Dice HD95 FgBgInstanceF1 ClassConfusion \
    --noise-scenarios clean roa roc noisy \
    --datasets LIDC RIGA Gleason MouseTumor MMIA MMIS \
    --datasets-for-metric Dice=LIDC,RIGA,Gleason,MouseTumor,MMIA,MMIS \
    --datasets-for-metric HD95=LIDC,RIGA,Gleason,MouseTumor,MMIA,MMIS \
    --datasets-for-metric FgBgInstanceF1=LIDC,RIGA,Gleason,MouseTumor,MMIA,MMIS \
    --datasets-for-metric ClassConfusion=LIDC,RIGA,Gleason,MouseTumor,MMIA,MMIS
```

The test uses the same final bootstrap-vector aggregation as the result tables
and ranking stability plots. It reports the original per-metric/per-scenario
dataset pairing, plus pooled pairings over dataset times scenario per metric.

#### Compile decision guide from ranking stability

Use the rank-frequency summaries produced by `visualize_ranking.py` together
with the ranking table from `ranking.py` to compile the FNLL decision guide.
The relevant generated artifacts are:

```text
results/segmentation_results/bootstrap_method_rankings.csv
results/segmentation_results/ranking_stability/rank_frequency_summary_<metric>_<datasets>.csv
```

The ranking stability plot shows how consistently each method ranks across
bootstrap resamples and noise scenarios; use these summaries to identify robust
method choices for each dataset and client-noise setting.

Additional comparison figures:

```bash
python3 ./src/eval/results_processing/partial_noisy_scenarios_global_comparison.py \
    --output-dir ./results/segmentation_results/partial_noise_comparison \
    --figure paired_dot
```

```bash
python3 ./src/eval/results_processing/roc_clean_vs_noisy_clients_global_comparison.py \
    --output-dir ./results/segmentation_results/partial_noise_comparison \
    --figure paired_dot
```

```bash
python3 ./src/eval/results_processing/robustness_analysis_noisy_scenarios_global.py \
    --figure separate_dot \
    --delta-mode abs
```

## Contribution guide

<details>
<summary>Incorporating a new FNLL method.</summary>

### Incorporation of a new FNLL method

This benchmark treats a federated noisy-label learning (FNLL) method as an FL
strategy. Existing examples live in `src/methods/` (`fedavg`, `feda3i`,
`feddm`, `fedcorr`, `fedselect`, `iopfl`) and are wired through
`src/fed/main.py`, `src/fed/orchestrator.py`, `src/fed/client.py`, and, if the
method changes the local training step, the nnU-Net trainer.

#### 1. Decide where your method acts

Most methods need one or more of these integration points:

- **Server-side aggregation only:** implement a custom aggregation function in
  `src/methods/<method>/<method>.py` and call it from
  `Orchestrator.aggregate`.
- **Client-side pre/post local training logic:** add a method-specific branch in
  `Client._run_strategy_round` or after `self.model.run(...)` in
  `Client.fed_round`.
- **Custom loss, sample weighting, label correction, or per-batch logic:** pass
  method flags/state through `src/fed/model.py` into
  `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`, then use them in
  `compute_training_loss`, `train_step`, validation, or dataloader setup.
- **Persistent method state for restarts:** store paths and hyperparameters in
  `self.fl_strategy_state` and implement `save_state`, following `IOPFL` or
  `FedCorr`.

#### 2. Add the method class

Create a new package under `src/methods/`, for example:

```text
src/methods/myfnll/
  myfnll.py
```

Start from `FedAvg` if the method still needs standard weighted averaging:

```python
from methods.fedavg.fedavg import FedAvg


class MyFNLL(FedAvg):
    def __init__(self, clients, myfnll_lambda=1.0, fl_strategy_state=None):
        super().__init__(clients)
        self.name = "myfnll"
        self.myfnll_lambda = (
            myfnll_lambda
            if fl_strategy_state is None
            else fl_strategy_state["myfnll_lambda"]
        )
        self.fl_strategy_state = {
            "myfnll_lambda": self.myfnll_lambda,
        }

    def myfnll_aggregate(self, client_checkpoints):
        # Return a state_dict-like dict containing the new server model weights.
        return self.fed_avg(client_checkpoints)

    def save_state(self, exp_id: str = None, client_id: int = None):
        self.save_fl_strategy_state_to_file(self.fl_strategy_state, exp_id)
```

The orchestrator expects strategy objects to expose `name`, `clients`, and any
method-specific functions called from the server or client flow.

#### 3. Register CLI arguments

In `src/fed/main.py`:

1. Add the method name and its hyperparameters to `METHOD_ARG_KEYS`.

```python
METHOD_ARG_KEYS = {
    ...
    "myfnll": ("myfnll_lambda",),
}
```

2. Add parser arguments near the other method arguments.

```python
parser.add_argument(
    "--myfnll_lambda",
    type=float,
    default=1.0,
    help="Regularization weight for MyFNLL.",
)
```

`build_fl_args` automatically copies all keys listed in `METHOD_ARG_KEYS` into
the orchestrator's `fl_args`.

#### 4. Build the strategy in the orchestrator

In `src/fed/orchestrator.py`, import the class:

```python
from methods.myfnll.myfnll import MyFNLL
```

Add it to `_build_fl_strategy`:

```python
if strategy_name == "myfnll":
    return MyFNLL(
        self.clients,
        fl_args["myfnll_lambda"],
        fl_strategy_state=fl_strategy_state,
    )
```

Add a server step in `_run_server_step`:

```python
elif strategy_name == "myfnll":
    self._run_myfnll_server_step(fl_round)
```

Then implement the step:

```python
def _run_myfnll_server_step(self, fl_round: int):
    self.aggregate(strategy="myfnll", fl_round=fl_round)
```

Finally, route the aggregation in `aggregate`:

```python
elif strategy == "myfnll":
    self.server_model_weights = self.fl_strategy.myfnll_aggregate(
        client_checkpoints
    )
```

If your method uses FedAvg unchanged, you can call `self.aggregate(strategy="fedavg")`
inside `_run_myfnll_server_step` instead.

#### 5. Add client-side hooks if needed

For methods that compute client statistics, select samples, maintain local
memory, or update personalized models, add a branch to
`Client._run_strategy_round`:

```python
elif strategy_name == "myfnll":
    self._run_myfnll_round(run_kwargs, fl_round, fl_strategy)
```

and implement:

```python
def _run_myfnll_round(self, run_kwargs: dict, fl_round: int, fl_strategy):
    run_kwargs.update(
        {
            "fl_client_id": self.client_id,
            "is_myfnll_active": True,
        }
    )
    self.model.run(**run_kwargs)
    fl_strategy.update_client_state(self.model.nnunet_trainer, self.client_id)
```

If the method only needs information after local training, follow the IOP-FL
pattern in `Client.fed_round`: call the strategy after `self.model.run(...)`
using `self.model.current_model_weights` or `self.model.nnunet_trainer`.

#### 6. Pass training-step flags through `src/fed/model.py`

If the nnU-Net trainer needs method-specific values, add them to both
`nnUNetv2_fed.run(...)` and `_build_run_training_kwargs(...)` in
`src/fed/model.py`, then include them in the returned kwargs passed to
`run_training`.

Example additions:

```python
def run(..., is_myfnll_active: bool = False):
    kwargs = self._build_run_training_kwargs(..., is_myfnll_active)
```

```python
return {
    ...
    "is_myfnll_active": is_myfnll_active,
}
```

Also make sure `Client._base_run_kwargs` or your method-specific client branch
sets the value.

#### 7. Create a method-specific nnU-Net trainer if needed

If your method changes the local training behavior, prefer a method-specific
trainer class over editing the base `nnUNetTrainer` directly. The benchmark
already follows this pattern in:

```text
nnUNet/nnunetv2/training/nnUNetTrainer/variants/fl/nnUNetTrainer_FL.py
```

That file contains simple aliases such as `nnUNetTrainer_FedAvg` and mixin-based
trainers such as `nnUNetTrainer_FedCorr`, `nnUNetTrainer_FedDM`, and
`nnUNetTrainer_FedSelect`.

Add your trainer to the same file, or create a new Python file below
`nnUNet/nnunetv2/training/nnUNetTrainer/variants/`. nnU-Net discovers trainers
by class name, so the class name you pass via `--trainer` must match the Python
class name.

Minimal example:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyFNLLTrainerMixin:
    def compute_training_loss(self, batch, data, output, target):
        loss = super().compute_training_loss(batch, data, output, target)
        if getattr(self.fl_strategy, "name", None) == "myfnll":
            loss = loss + self.fl_strategy.myfnll_regularizer(
                batch=batch,
                output=output,
                target=target,
                trainer=self,
            )
        return loss


class nnUNetTrainer_MyFNLL(MyFNLLTrainerMixin, nnUNetTrainer):
    pass
```

If you use a custom base trainer, such as `nnUNetTrainerDiceCELoss_noSmooth`,
compose the mixin with that base instead:

```python
class nnUNetTrainerDiceCELoss_noSmooth_MyFNLL(
    MyFNLLTrainerMixin,
    nnUNetTrainerDiceCELoss_noSmooth,
):
    pass
```

Then launch the benchmark with the new trainer:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method myfnll \
    --trainer nnUNetTrainer_MyFNLL \
    ...
```

Use this trainer-subclass route for changes to `_build_loss`,
`compute_training_loss`, `train_step`, `run_train_iterations`, dataloaders,
augmentation, validation behavior, or any method-specific local training state.
Use the strategy class in `src/methods/<method>/` for server aggregation and
state that belongs to the FL algorithm.

#### 8. Use method flags inside nnU-Net training if needed

For loss or per-batch behavior, extend your method-specific trainer class:

1. Add constructor arguments with defaults, for example
   `is_myfnll_active: bool = False`.
2. Store them as instance attributes near the existing FL args.
3. Use the attributes in `compute_training_loss` or `train_step`.

Example:

```python
def compute_training_loss(self, batch, data, output, target):
    loss = self.loss(output, target)
    if self.is_myfnll_active:
        loss = loss + self.fl_strategy.myfnll_regularizer(
            batch=batch,
            output=output,
            target=target,
            trainer=self,
        )
    return loss
```

Keep tensor operations on `self.device`, avoid storing GPU tensors in long-lived
strategy state unless necessary, and move persistent state to CPU before saving
when possible.

#### 9. Save and restart method state

If your method has state that must survive restarts, keep JSON-serializable
metadata in `self.fl_strategy_state`. Save large tensors or model weights as
separate `.pth` files and store only their paths in the JSON. `IOPFL.save_state`
is the reference pattern for per-client tensor checkpoints, while
`FedCorr.save_global_model_weights` is the reference pattern for global model
state.

Restart support is driven by the `fl_strategy_state` entry in the experiment
args JSON. In your method constructor, accept `fl_strategy_state=None` and load
saved values from it when present.

#### 10. Run a small smoke test

Before launching a full benchmark, run a tiny experiment with a few rounds and
one local epoch:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method myfnll \
    --dataset_ids "505 506 507 508" \
    --num_clients 4 \
    --num_rounds 2 \
    --num_local_epochs 1 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_MyFNLL \
    --myfnll_lambda 1.0
```

Check that:

- the method name appears in the generated `ExperimentArgs_*.json`;
- local training finishes for every client;
- `Orchestrator.aggregate` produces `server_model_weights`;
- final checkpoints are written in each client result folder;
- any method-specific state can be saved and loaded again by the restart script.

</details>


<details>
<summary>Incorporating a new dataset with segmentation label noise.</summary>

### Incorporation of a new dataset

New datasets should enter the benchmark through the nnU-Net dataset interface.
Keep raw data, preprocessed data, and experiment results outside the repository
and point the suite to them with the standard environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

#### 1. Convert the dataset to nnU-Net format

Create a `DatasetXXX_<Name>` folder under `nnUNet_raw` with the standard
`imagesTr`, `labelsTr`, and `dataset.json` layout. Use a unique dataset ID for
each client dataset that participates in FL. If you add noisy labels, keep the
clean reference and noisy labels in clearly named folders so the benchmark CLI
can select them via `--clean_validation_dataset` and `--noisy_train_folder`.

#### 2. Check label-noise metadata and splits

Make sure each client dataset exposes the same label set, image channels, and
compatible train/validation splits. FL aggregation assumes all clients train the
same model architecture, so mismatched labels, modalities, or planning outputs
will break aggregation.

#### 3. Run federated planning and preprocessing

Use `src/data/utils/nnunet_fed_preparation.py` across all client dataset IDs.
This computes client fingerprints, averages them centrally, and writes common
plans so all clients use compatible network weights.

```bash
python3 ./src/data/utils/nnunet_fed_preparation.py \
    --dataset_ids "505 506 507 508" \
    --configuration "3d_fullres" \
    --planner "nnUNetPlannerResEncM" \
    --plans_name "nnUNetResEncUNetMPlans" \
    --verify_dataset_integrity
```

#### 4. Smoke-test the dataset in FL

Run a short FedAvg experiment before evaluating FNLL methods:

```bash
python3 ./src/fed/main.py \
    --noise_mitigation_method fedavg \
    --dataset_ids "505 506 507 508" \
    --num_clients 4 \
    --num_rounds 2 \
    --num_local_epochs 1 \
    --configuration 3d_fullres \
    --plan nnUNetResEncUNetMPlans \
    --trainer nnUNetTrainer_FedAvg
```

Check that every client trains, aggregation finishes, validation runs, and the
result folders are created below `nnUNet_results`.

</details>
