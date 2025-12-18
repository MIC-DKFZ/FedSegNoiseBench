import os
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
        save_every: int = 50,
        num_gpus: int = 1,
        continue_training: bool = False,
        clean_validation_dataset: str = None,
        experiment_id: str = None,
        noisy_train_folder: str = None,
        noise_ratio: float = None,
    ):
        # set input args
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.plan = plan
        self.trainer = trainer
        self.save_every = save_every
        self.clean_validation_folder = (
            os.path.join(
                os.getenv("nnUNet_preprocessed"),
                clean_validation_dataset,
                f"nnUNetPlans_{self.configuration}",
            )
            if clean_validation_dataset
            else None
        )
        self.experiment_id = experiment_id
        self.noisy_train_folder = (
            os.path.join(
                os.getenv("nnUNet_preprocessed"),
                noisy_train_folder,
                f"nnUNetPlans_{self.configuration}",
            )
            if noisy_train_folder
            else None
        )
        self.noise_ratio = noise_ratio

        # initate model
        self.current_model_weights = None
        self.nnunet_trainer = None
        self.run(
            initialize_fed_training=True,
            continue_training=continue_training,
            num_gpus=num_gpus,
        )

    def run(
        self,
        num_gpus: int = 1,
        initialize_fed_training: bool = False,
        continue_training: bool = False,
        num_epochs: int = 1000,
        current_epoch: int = 0,
        epochs_per_round: int = 1,
        last_fl_round: bool = False,
        very_last_fl_predict_round: bool = False,
        only_run_validation: bool = False,
        fl_strategy=None,
        fl_client_id: int = None,
        feddm_client_peers: list = None,
        is_fedcorr_noisyclient: bool = False,
        is_fedcorr_preproc_stage: bool = False,
        is_fedcorr_finetune_stage: bool = False,
        is_fedcorr_fulltrain_stage: bool = False,
    ):
        self.current_model_weights, self.nnunet_trainer = run_training(
            nnunet_trainer=self.nnunet_trainer,
            dataset_name_or_id=self.dataset_id,
            configuration=self.configuration,
            fold=self.fold,
            trainer_class_name=self.trainer,
            plans_identifier=self.plan,
            pretrained_weights=None,
            num_gpus=num_gpus,
            export_validation_probabilities=False,
            continue_training=continue_training,
            only_run_validation=only_run_validation,
            disable_checkpointing=False,
            val_with_best=False,
            device=torch.device("cuda"),
            save_every=self.save_every,
            clean_validation_folder=self.clean_validation_folder,
            initialize_fed_training=initialize_fed_training,
            num_epochs=num_epochs,
            current_epoch=current_epoch,
            epochs_per_round=epochs_per_round,
            last_fl_round=last_fl_round,
            current_model_weights=self.current_model_weights,
            very_last_fl_predict_round=very_last_fl_predict_round,
            experiment_id=self.experiment_id,
            fl_strategy=fl_strategy,
            fl_client_id=fl_client_id,
            feddm_client_peers=feddm_client_peers,
            noisy_train_folder=self.noisy_train_folder,
            noise_ratio=self.noise_ratio,
            is_fedcorr_noisyclient=is_fedcorr_noisyclient,
            is_fedcorr_preproc_stage=is_fedcorr_preproc_stage,
            is_fedcorr_finetune_stage=is_fedcorr_finetune_stage,
            is_fedcorr_fulltrain_stage=is_fedcorr_fulltrain_stage,
        )
