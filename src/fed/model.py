import torch

from nnUNet.nnunetv2.run.run_training import run_training


class nnUNetv2_fed:
    def __init___(
        self, dataset_id, configuration, fold, plan, trainer, clean_validation_folder
    ):
        # set input args
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.plan = plan
        self.trainer = trainer
        self.clean_validation_folder = clean_validation_folder

        # initate model
        self.model_checkpoint = self.run(initialize_fed_training=True)

    def run(self, initialize_fed_training=False):
        run_training(
            dataset_name_or_id=self.dataset_id,
            configuration=self.configuration,
            fold=self.fold,
            trainer_class_name=self.trainer,
            plans_identifier=self.plan,
            pretrained_weights=None,
            num_gpus=1,
            export_validation_probabilities=False,
            continue_training=False,
            only_run_validation=False,
            disable_checkpointing=False,
            val_with_best=False,
            device=torch.device("cuda"),
            clean_validation_folder=self.clean_validation_folder,
            initialize_fed_training=initialize_fed_training,
        )
