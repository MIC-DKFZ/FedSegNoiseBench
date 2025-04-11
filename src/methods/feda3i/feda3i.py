import logging
from tqdm import tqdm
import copy

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.ndimage.morphology import distance_transform_edt as distrans

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.fedavg.fedavg import FedAvg


class FedA3I(FedAvg):
    def __init__(self, clients: list = None, feda3i_warmup_rounds: int = None):
        super().__init__(clients=clients)
        self.name = "feda3i"
        self.feda3i_warmup_rounds = feda3i_warmup_rounds
        self.feda3i_interw = 0.5
        self.quality_agg_weights = None
        self.alpha_weight = 1.0
        self.alpha_power = 1.0
        self.alpha_bias = 0.0

    def feda3i_compute_quality_agg_weights(self):
        """
        Estimate client's noise and compute quality-based aggregation weights.

        Notes:
        - pretty much copied and adapted from https://github.com/wnn2000/FedAAAI/blob/main/code/train_FedAAAI.py
        """
        # collect loss values
        loss_client = np.zeros(
            (len(self.clients), 2 * (len(self.clients[0].dataset_json["labels"]) - 1))
        )
        for client_idx, client in enumerate(self.clients):
            # compute two-directional loss
            loss_n = self.cal_loss_two_directions(
                clients_nnunet_trainer=client.model.nnunet_trainer
            )
            loss_client[client_idx] = np.nanmean(loss_n, axis=0)
        logging.info(f"Clients losses: {loss_client}")

        # noise classification
        # original code with GaussianMixture(n_components=2, covariance_type="full") but causes error "Segmentation fault (core dumped)"
        gmm = GaussianMixture(
            n_components=2, covariance_type="diag", verbose=1, verbose_interval=1
        ).fit(loss_client)
        gmm_pred = gmm.predict(loss_client)
        odd0 = np.sum(gmm.means_[0][1::2])
        even0 = np.sum(gmm.means_[0][::2])
        odd1 = np.sum(gmm.means_[1][1::2])
        even1 = np.sum(gmm.means_[1][::2])
        y_expand = 0 if (even0 - odd0) > (even1 - odd1) else 1
        gmm_ex = np.where(gmm_pred == y_expand)[0]
        gmm_sh = np.where(gmm_pred == (1 - y_expand))[0]
        logging.info(f"=====> selected expand clients: {gmm_ex}")
        logging.info(f"=====> selected shrink clients: {gmm_sh}")
        loss_ex = loss_client[gmm_ex][:, 0] - loss_client[gmm_ex][:, 1]
        loss_sh = loss_client[gmm_sh][:, 1] - loss_client[gmm_sh][:, 0]

        # quality-based weight calculation
        eps = 1e-4  # avoid division by zero and NaNs
        self.quality_agg_weights = np.zeros(len(self.clients))
        self.quality_agg_weights[gmm_ex] = 1 - (loss_ex - (loss_ex.min() - eps)) / (
            (loss_ex.max() + eps) - (loss_ex.min() - eps)
        )
        self.quality_agg_weights[gmm_ex] = (
            self.feda3i_interw
            * self.quality_agg_weights[gmm_ex]
            / self.quality_agg_weights[gmm_ex].sum()
        )
        self.quality_agg_weights[gmm_sh] = 1 - (loss_sh - (loss_sh.min() - eps)) / (
            (loss_sh.max() + eps) - (loss_sh.min() - eps)
        )
        self.quality_agg_weights[gmm_sh] = (
            (1 - self.feda3i_interw)
            * self.quality_agg_weights[gmm_sh]
            / self.quality_agg_weights[gmm_sh].sum()
        )
        logging.info(f"=====> Quality-based weights: {self.quality_agg_weights}")

    def feda3i_aggregate(self, client_checkpoints: dict = None):
        """
        FedA3I's quality- and quantity-based, layerwise aggregation.
        Notes:
        - Code from: https://github.com/wnn2000/FedAAAI/blob/main/code/utils/FedAvg.py#L16
        - Weight1-weight2-weighted aka. quality-quantity-weighted FedAvg highly adapted from FedAvg.fed_avg() method
        """
        # quality-based weights
        weight1 = self.quality_agg_weights
        # quantity-based weights
        weight2 = np.array(
            [
                (
                    self.get_num_datasamples_client(client.client_id)
                    / self.get_num_datasamples_client()
                )
                for client in self.clients
            ]
        )
        assert np.allclose(np.sum(weight1), 1), f"{np.sum(weight1)} does not sum to 1.0"
        assert np.allclose(np.sum(weight2), 1), f"{np.sum(weight2)} does not sum to 1.0"
        w_avg = copy.deepcopy(client_checkpoints[0])

        # create deepcopy of client model weights to not directly modify them
        _client_checkpoints = copy.deepcopy(client_checkpoints)

        keys = list(w_avg.keys())
        address_key_dict = {}
        for k in keys:
            address = w_avg[k].data_ptr()
            if address not in address_key_dict.keys():
                address_key_dict[address] = [k]
            else:
                address_key_dict[address].append(k)

        # Define stages and their corresponding temp values
        address_temp_mapping = {}
        # Define the stage keywords in depth order
        stage_keywords_in_order = [
            "encoder.stem.convs",
            "encoder.stages.0",
            "encoder.stages.1",
            "encoder.stages.2",
            "encoder.stages.3",
            "encoder.stages.4",
            "encoder.stages.5",
            "encoder.stages.6",
            "encoder.stages.7",
            "decoder.stages.0",
            "decoder.stages.1",
            "decoder.stages.2",
            "decoder.stages.3",
            "decoder.stages.4",
            "decoder.stages.5",
            "decoder.stages.6",
            "decoder.transpconvs",
            "decoder.seg_layers",
        ]
        # Get all unique keys from the model (weights)
        all_keys = list(w_avg.keys())
        # Find which stages are actually present
        present_stages = []
        for stage in stage_keywords_in_order:
            if any(stage in key for key in all_keys):
                present_stages.append(stage)
        # Dynamically assign temperatures
        stage_keywords_temp_dict = {
            stage: temp for temp, stage in enumerate(present_stages)
        }
        # assign temp values to addresses
        for address, keys in address_key_dict.items():
            temp = None
            for i, (stage, temp) in enumerate(stage_keywords_temp_dict.items()):
                if keys[0].startswith(stage):
                    address_temp_mapping[address] = temp
                    break

        # Compute alpha to weight weight1 and weight2 based on temp values, i.e. model depth
        alpha = np.linspace(0.0, 1.0, len(stage_keywords_temp_dict))
        alpha = self.alpha_weight * np.power(alpha, self.alpha_power) + self.alpha_bias

        # perform the weight1-weight2-weighted fedavg in a layer-wise manner
        for a, keys in address_key_dict.items():
            # compute weight3
            weight3 = (
                alpha[address_temp_mapping[a]] * weight1
                + (1 - alpha[address_temp_mapping[a]]) * weight2
            )

            for client_id, client_model_weights in _client_checkpoints.items():
                if client_id == "0" or client_id == 0:
                    # network weights of client_id="0" are already in _server_model_weights
                    # we still need to weight client 0's model params with it's dataset size
                    w_avg[address_key_dict[a][0]] = (
                        w_avg[address_key_dict[a][0]]
                        # * self.get_num_datasamples_client(client_id)
                        * weight3[client_id]
                    )
                else:
                    # weighted sum
                    w_avg[address_key_dict[a][0]] += (
                        client_model_weights[address_key_dict[a][0]]
                        # * self.get_num_datasamples_client(client_id)
                        * weight3[client_id]
                    )

        return w_avg

    def cal_loss_two_directions(self, clients_nnunet_trainer):
        # get net, loader, criterion from nnunet_trainer
        net = clients_nnunet_trainer.network.cuda()
        loader = clients_nnunet_trainer.dataloader_train
        # criterion = clients_nnunet_trainer.loss
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        # from here basically FedA3I code (https://github.com/wnn2000/FedAAAI/blob/main/code/utils/utils.py#L80)
        net.eval()
        with torch.no_grad():
            # for i, batch in enumerate(tqdm(loader, desc="Processing batches")):
            num_batches = len(loader.generator._data.identifiers) // len(
                loader.generator.get_indices()
            )
            for i in tqdm(range(num_batches), desc="Processing Batches"):
                batch = next(loader)
                images = batch["data"].cuda()
                labels = [x.cuda() for x in batch["target"]]
                outputs = net(images)
                if criterion is None:
                    raise
                else:
                    # get outputs and labels of highest resolution, as labels are in nnUNet by default a list
                    # e.g. [[b, c, h, w], [b, c, h/2, w/2], ...] for 2D due to deep supervision of nnUNet
                    high_res_outputs, high_res_labels = outputs[0], labels[0]

                    # num_classes: count of fg classes + bg class
                    num_classes = int(high_res_labels.max()) + 1
                    # get labels as one-hot encoded labels
                    # one_hot_labels to shape: [b, (d), h, w, c]
                    one_hot_labels = F.one_hot(
                        high_res_labels.squeeze(1).long(), num_classes=num_classes
                    )
                    # one_hot_labels to shape: [b, c, (d), h, w]
                    if len(one_hot_labels.shape) == 4:  # for 2D
                        one_hot_labels = one_hot_labels.permute(0, 3, 1, 2).float()
                    elif len(one_hot_labels.shape) == 5:  # for 3D
                        one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3).float()
                    else:
                        raise ValueError(
                            f"Unexpected shape of one_hot_labels: {one_hot_labels.shape}"
                        )
                    loss = criterion(
                        high_res_outputs, one_hot_labels
                    )  # [b, c, (d), h, w]
                    loss_feature = torch.zeros(
                        (
                            one_hot_labels.shape[0],
                            2 * (num_classes - 1),
                        )
                    )
                    # iterate over foreground classes
                    for c in range((num_classes - 1)):
                        region_mask = torch.from_numpy(
                            self.region(one_hot_labels[:, (c + 1)].unsqueeze(1))
                        ).cuda()
                        assert (region_mask == 0).any()
                        loss_n = loss[:, c].unsqueeze(1)
                        assert region_mask.shape == loss_n.shape
                        loss_n_in = (loss_n * (region_mask == 1).float()).view(
                            loss_n.shape[0], -1
                        ).sum(1) / (region_mask == 1).float().view(
                            loss_n.shape[0], -1
                        ).sum(
                            1
                        )
                        loss_n_out = (loss_n * (region_mask == 2).float()).view(
                            loss_n.shape[0], -1
                        ).sum(1) / (region_mask == 2).float().view(
                            loss_n.shape[0], -1
                        ).sum(
                            1
                        )
                        assert (
                            loss_n_in.shape[0] == loss_n_out.shape[0] == images.shape[0]
                        )
                        loss_feature[:, c * 2] = loss_n_in
                        loss_feature[:, c * 2 + 1] = loss_n_out

                if i == 0:
                    loss_whole_n = loss_feature.cpu().numpy()
                else:
                    loss_whole_n = np.concatenate(
                        (loss_whole_n, loss_feature.cpu().numpy()), axis=0
                    )
        return loss_whole_n

    def region(self, labels):
        """
        Unchanged FedA3I code: https://github.com/wnn2000/FedAAAI/blob/main/code/utils/utils.py#L56
        """
        labels = labels.cpu().numpy().astype("float")
        sdm = np.zeros_like(labels).astype("float")
        region_mask = np.zeros_like(labels).astype("float")
        # print(labels.dtype, sdm.dtype, region_mask.dtype)
        for i in range(labels.shape[0]):
            if labels[i].sum() < 1:
                pass
            else:
                pos_dis = distrans(labels[i])
                neg_dis = distrans(1 - labels[i])
                sdm[i] = -pos_dis + neg_dis
                min_dis = sdm[i].min()
                max_dis = sdm[i].max()
                assert min_dis < 0
                assert max_dis > 0
                dis = -min_dis if -min_dis < max_dis else max_dis
                assert dis > 0, f"{dis} is not bigger than 0"
                region_mask[i][(labels[i] == 0) & (sdm[i] <= dis)] = 2
                region_mask[i][(labels[i] == 1) & (sdm[i] >= -dis)] = 1

        return region_mask


if __name__ == "__main__":
    # a = np.array([[ 0.25016564, 15.4473114,   5.21507978,  0.36437854], [ 0.22284667,  6.31971169,  1.45446646,  0.74241704], [ 0.07514828, 17.83001137,  5.70706367,  0.28784826]])
    # gmm = GaussianMixture(n_components=2, verbose=1, verbose_interval=1).fit(a)
    # gmm_pred = gmm.predict(a)
    print("done")
