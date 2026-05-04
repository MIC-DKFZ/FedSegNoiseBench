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

## Adding a new FNLL method

This benchmark treats a federated noisy-label learning (FNLL) method as an FL
strategy. Existing examples live in `src/methods/` (`fedavg`, `feda3i`,
`feddm`, `fedcorr`, `fedselect`, `iopfl`) and are wired through
`src/fed/main.py`, `src/fed/orchestrator.py`, `src/fed/client.py`, and, if the
method changes the local training step, the nnU-Net trainer.

### 1. Decide where your method acts

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

### 2. Add the method class

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

### 3. Register CLI arguments

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

### 4. Build the strategy in the orchestrator

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

### 5. Add client-side hooks if needed

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

### 6. Pass training-step flags through `src/fed/model.py`

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

### 7. Create a method-specific nnU-Net trainer if needed

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

### 8. Use method flags inside nnU-Net training if needed

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

### 9. Save and restart method state

If your method has state that must survive restarts, keep JSON-serializable
metadata in `self.fl_strategy_state`. Save large tensors or model weights as
separate `.pth` files and store only their paths in the JSON. `IOPFL.save_state`
is the reference pattern for per-client tensor checkpoints, while
`FedCorr.save_global_model_weights` is the reference pattern for global model
state.

Restart support is driven by the `fl_strategy_state` entry in the experiment
args JSON. In your method constructor, accept `fl_strategy_state=None` and load
saved values from it when present.

### 10. Run a small smoke test

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
python3 ./src/data/data_analysis/visualize_hd95_f1_confusion.py \
    --json_path /results/noise_analysis/noise_analysis_results_clean<DATASET-IDS-OF-YOUR-DATASET>.json \
    --output ./results/noise_analysis/<YOUR-DATASET>/hd95_vs_f1_vs_confusion.png \
    --level instance \
    --figsize 12.4 10.8
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
python3 ./src/eval/results_processing/visualize_results.py --metric Dice
```

Overall ranking stability plots:
```
python3 ./src/eval/results_processing/visualize_ranking.py --output-dir ./results/segmentation_results/ranking_stability --metric Dice
```

Overall partial noise settings (`roa` vs `roc`) comparison:
```
python3 ./src/eval/results_processing/partial_noisy_scenarios_global_comparison.py --output-dir ./results/segmentation_results/partial_noise_comparison --figure paired_dot
```

Per-client -- roc-clean vs roc-noisy clients comparison:
```
python3 ./src/eval/results_processing/roc_clean_vs_noisy_clients_global_comparison.py --output-dir ./results/segmentation_results/partial_noise_comparison --figure paired_dot
```

Overall clean-referenced robustness analysis:
```
python ./src/eval/results_processing/robustness_analysis_noisy_scenarios_global.py --figure separate_dot --delta-mode abs
```
