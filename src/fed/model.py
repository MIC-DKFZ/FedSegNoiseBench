import torch

from nnunetv2.run.run_training import run_training


class nnUNetv2_fed:
    def __init__(
        self,
        dataset_id: int = None,
        configuration: str = None,
        fold: int = None,
        plan: str = None,
        trainer: str = None,
        clean_validation_folder: str = None,
    ):
        # set input args
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.plan = plan
        self.trainer = trainer
        self.clean_validation_folder = clean_validation_folder

        # initate model
        self.current_model_weights = None
        self.model_checkpoint = self.run(initialize_fed_training=True)

    def run(
        self,
        initialize_fed_training: bool = False,
        continue_training: bool = False,
        num_epochs: int = 1000,
        current_epoch: int = 0,
        epochs_per_round: int = 1,
        last_fl_round: bool = False,
    ):
        self.current_model_weights = run_training(
            dataset_name_or_id=self.dataset_id,
            configuration=self.configuration,
            fold=self.fold,
            trainer_class_name=self.trainer,
            plans_identifier=self.plan,
            pretrained_weights=None,
            num_gpus=1,
            export_validation_probabilities=False,
            continue_training=continue_training,
            only_run_validation=False,
            disable_checkpointing=False,
            val_with_best=False,
            device=torch.device("cuda"),
            clean_validation_folder=self.clean_validation_folder,
            initialize_fed_training=initialize_fed_training,
            num_epochs=num_epochs,
            current_epoch=current_epoch,
            epochs_per_round=epochs_per_round,
            last_fl_round=last_fl_round,
            current_model_weights=self.current_model_weights,
        )
