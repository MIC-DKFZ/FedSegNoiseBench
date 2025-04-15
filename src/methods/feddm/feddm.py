import argparse
import warnings
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Dict, List, Tuple, Optional, cast

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils import (
    map_,
    depth,
    str2bool,
    inter_sum,
    union_sum,
    probs2one_hot,
    dice_coef,
    iIoU,
)
from losses import Focal_Cross_Entropy as focal_cross_entropy

from methods.fedavg.fedavg import FedAvg


class FedDM(FedAvg):
    """
    FedDM: Federated Weakly Supervised Segmentation via Annotation Calibration and Gradient De-Conflicting
    Meilu Zhu et al., 2023, IEEE Transactions on Medical Imaging
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10013742&tag=1
    """

    def __init__(self, clients: list = None):
        super().__init__(clients=clients)
        self.name = "feddm"

    def initialize_grad_len(self, server_model, grad_history):
        """
        Initialize gradient length for each key in the grad_history.
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L608C1-L616C20
        """
        grad_len = {key: 0 for key in grad_history.keys()}
        for g_key in grad_len.keys():
            for key in server_model.state_dict().keys():
                if g_key in key:
                    dims = server_model.state_dict()[key].shape
                    grad_len[g_key] += dims.numel()
        return grad_len

    def do_epoch_peer(
        self,
        args,
        mode: str,
        net: Any,
        device: Any,
        loader: DataLoader,
        epc: int,
        list_loss_fns: List[List[Callable]],
        K: int,
        savedir: str = "",
        optimizer: Any = None,
        compute_miou: bool = False,
        temperature: float = 1,
        client_idx=None,
        lr=None,
        peer_models=None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L207
        """
        assert mode in ["train", "val"]

        if mode == "train":
            net.train()
        elif mode == "val":
            net.eval()

        total_iteration: int = len(loader)  # U
        total_images: int = len(loader.dataset)
        n_loss: int = max(map(len, list_loss_fns))

        all_dices: Tensor = torch.zeros(
            (total_images, K), dtype=torch.float32, device=device
        )
        loss_log: Tensor = torch.zeros(
            (total_iteration, n_loss), dtype=torch.float32, device=device
        )

        iiou_log: Optional[Tensor]
        intersections: Optional[Tensor]
        unions: Optional[Tensor]
        if compute_miou:
            iiou_log = torch.zeros(
                (total_images, K), dtype=torch.float32, device=device
            )
            intersections = torch.zeros(
                (total_images, K), dtype=torch.float32, device=device
            )
            unions = torch.zeros((total_images, K), dtype=torch.float32, device=device)
        else:
            iiou_log = None
            intersections = None
            unions = None

        done_img: int = 0
        done_batch: int = 0
        loss_fns = list_loss_fns[0]

        peer_model_nearst = peer_models[0].eval()
        peer_model_farthest = peer_models[1].eval()

        n_epoch = args.stop_epoch
        ratio = args.ratio
        p = 1 - (ratio * epc / n_epoch)
        if epc > n_epoch:
            p = 1 - ratio

        seg_sen = []
        seg_spe = []
        seg_acc = []
        seg_jac_score = []

        for data in loader:
            image: Tensor = data["images"].to(device)
            target: Tensor = data["gt"].to(device)
            assert not target.requires_grad
            labels: List[Tensor] = [e.to(device) for e in data["labels"]]
            # meilu
            B, C, *_ = image.shape
            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

                # Forward
            pred_logits = net(image)

            with torch.no_grad():
                pred_logits1 = peer_model_nearst(image)
                pred_logits2 = peer_model_farthest(image)

            clean_mask = self.pixel_selection_by_Peers(
                pred_logits.detach(),
                pred_logits1.detach(),
                pred_logits2.detach(),
                labels,
                p=p,
            )

            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())
            assert not predicted_mask.requires_grad

            mask = target[:, 1, :, :].cpu().data.numpy()
            pred_segs = pred_probs.cpu().data.numpy()
            smooth: float = 1e-8
            for i in range(B):
                val_mask = mask[i]
                y_true_f = val_mask.reshape(
                    val_mask.shape[0] * val_mask.shape[1], order="F"
                )
                pred_seg = pred_segs[i]
                pred_arg = np.argmax(pred_seg, axis=0)
                y_pred_f = pred_arg.reshape(
                    pred_arg.shape[0] * pred_arg.shape[1], order="F"
                )
                intersection = np.float(np.sum(y_true_f * y_pred_f))
                seg_sen.append((intersection + smooth) / (np.sum(y_true_f) + smooth))
                intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
                seg_spe.append(
                    (intersection0 + smooth) / (np.sum(1 - y_true_f) + smooth)
                )
                seg_acc.append(accuracy_score(y_true_f, y_pred_f))
                seg_jac_score.append(
                    (intersection + smooth)
                    / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)
                )

            mask_receptacle = predicted_mask[...]

            loss1 = focal_cross_entropy(pred_probs, labels[0], clean_mask)
            losses = [loss1]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()

            # Compute and log metrics
            loss_sub_log: Tensor = torch.zeros(
                len(loss_fns), dtype=torch.float32, device=device
            )
            for j in range(len(loss_fns)):
                loss_sub_log[j] = losses[j].detach()
            loss_log[done_batch, ...] = loss_sub_log[...]
            del loss_sub_log

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(mask_receptacle, target)
            assert dices.shape == (B, K), (dices.shape, B, K)
            all_dices[sm_slice, ...] = dices

            if compute_miou:
                IoUs: Tensor = iIoU(mask_receptacle, target)
                assert IoUs.shape == (B, K), IoUs.shape
                iiou_log[sm_slice] = IoUs  # type: ignore
                intersections[sm_slice] = inter_sum(mask_receptacle, target)  # type: ignore
                unions[sm_slice] = union_sum(mask_receptacle, target)  # type: ignore

            # Logging
            done_img += B
            done_batch += 1

        mIoUs: Optional[Tensor]
        if intersections is not None and unions is not None:
            mIoUs = intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)
            assert mIoUs.shape == (K,), mIoUs.shape
        else:
            mIoUs = None

        loss = loss_log.mean().detach().cpu()
        DSC = all_dices.mean().detach().cpu()
        DSC0 = all_dices[:, 0].mean().detach().cpu()
        DSC1 = all_dices[:, 1].mean().detach().cpu()
        mIoU = mIoUs.mean().detach().cpu()

        seg_sen = np.nanmean(seg_sen)
        seg_spe = np.nanmean(seg_spe)
        seg_acc = np.nanmean(seg_acc)
        seg_jac_score = np.nanmean(seg_jac_score)

        return loss, DSC, DSC0, DSC1, mIoU, seg_sen, seg_spe, seg_acc, seg_jac_score

    def pixel_selection_by_Peers(self, logits, logits1, logits2, labels, p=0):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L356
        """
        bg_mask = labels[0][:, 0, :, :]  # B, H, W
        fg_mask = labels[0][:, 1, :, :]  # B, H, W

        pred = torch.softmax(logits, dim=1)  # B, 2, H, W
        pred1 = torch.softmax(logits1, dim=1)  # B, 2, H, W
        pred2 = torch.softmax(logits2, dim=1)  # B, 2, H, W
        log_p: Tensor = (pred + 1e-10).log()
        log_p1: Tensor = (pred1 + 1e-10).log()
        log_p2: Tensor = (pred2 + 1e-10).log()
        mask: Tensor = cast(Tensor, labels[0].type(torch.float32))
        loss = -mask * log_p  # ls B, 2, H, W
        loss_fg = loss[:, 1, :, :]
        loss_fg_flatten = loss_fg.flatten(1, 2)

        loss1 = -mask * log_p1  # B, 2, H, W
        loss1_fg = loss1[:, 1, :, :]
        loss1_fg_flatten = loss1_fg.flatten(1, 2)

        loss2 = -mask * log_p2  # B, 2, H, W
        loss2_fg = loss2[:, 1, :, :]
        loss2_fg_flatten = loss2_fg.flatten(1, 2)

        clean_mask = torch.zeros_like(loss_fg)  # B, H, W

        for b in range(fg_mask.size(0)):

            # fg_num = (fg_mask.sum((1,2)) * p).type(torch.int)
            fg_num_selected = (fg_mask[b].sum() * p).type(torch.int).item()
            threshold = fg_num_selected + bg_mask[b].sum()
            # print('fg_num:', fg_num_selected)
            if fg_num_selected > 5:
                value_fg, _ = torch.topk(
                    loss_fg_flatten[b, :], threshold, largest=False, sorted=True
                )
                thresh_fg = value_fg[-1]
                value_fg1, _ = torch.topk(
                    loss1_fg_flatten[b, :], threshold, largest=False, sorted=True
                )
                thresh_fg1 = value_fg1[-1]
                value_fg2, _ = torch.topk(
                    loss2_fg_flatten[b, :], threshold, largest=False, sorted=True
                )
                thresh_fg2 = value_fg2[-1]
                clean_mask_ = loss_fg[b, :, :] <= thresh_fg
                clean_mask1_ = loss1_fg[b, :, :] <= thresh_fg1
                clean_mask2_ = loss2_fg[b, :, :] <= thresh_fg2

                clean_mask[b, :, :][(clean_mask_ & clean_mask2_)] = 1.0
                clean_mask[b, :, :][
                    (clean_mask_ | clean_mask1_) ^ (clean_mask_ & clean_mask1_)
                ] = 2
            else:
                clean_mask[b, :, :] = 1.0

        clean_mask = clean_mask * fg_mask + bg_mask  # B, H, W

        return clean_mask

    def communication_FedDM(
        self,
        args,
        server_model,
        models,
        client_weights,
        device=None,
        gauss=None,
        embeddings=None,
        epoch=None,
        grad_history=None,
    ):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L578
        """

        peer_models, embeddings, nearest_clients_bulk, farthest_clients_bulk = (
            self.find_customized_peers(models, gauss, embeddings, device)
        )

        grads = []
        for model in models:
            grads.append(self.get_grads_(model, server_model))

        new_grads, grad_history = self.pcgrad_hierarchy(args, grads, grad_history)

        for k, model in enumerate(models):
            models[k] = self.set_grads_(model, server_model, new_grads)

        with torch.no_grad():
            # aggregate params
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(
                        models[0].state_dict()[key]
                    )
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

        return server_model, models, peer_models, embeddings, grad_history

    def find_customized_peers(self, models, input, embeddings, device):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L419
        """
        customized_peers = []
        for client_idx, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                # increase the sampling size by batch processing
                for i in range(4):
                    input_ = input[
                        i * input.size(0) // 4 : (i + 1) * input.size(0) // 4
                    ]
                    out = torch.softmax(model(input_), dim=1)  # 100, 2, 256, 256
                    embeddings[client_idx][
                        i * input.size(0) // 4 : (i + 1) * input.size(0) // 4
                    ] = out

        nearest_clients_bulk = torch.zeros(len(embeddings))
        farthest_clients_bulk = torch.zeros(len(embeddings))
        for client_i in range(len(embeddings)):
            embedding = embeddings[client_i].reshape(
                embeddings[client_i].size(0), -1
            )  # 100, 2*256*256
            nearest_samples_bulk = torch.zeros(len(embeddings))
            farthest_samples_bulk = torch.zeros(len(embeddings))
            for b in range(embedding.size(0)):
                distances = torch.zeros(len(embeddings))
                for client_j in range(len(embeddings)):
                    if client_i == client_j:
                        distances[client_j] = 1.0
                    else:
                        embedding_o = embeddings[client_j][b].view(-1)
                        distances[client_j] = torch.norm(
                            embedding[b] - embedding_o, p=2
                        )

                distances[client_i] = 1e10
                nearest_idx = distances.argmin()
                nearest_samples_bulk[nearest_idx] += 1
                distances[client_i] = -1e10
                farthest_idx = distances.argmax()
                farthest_samples_bulk[farthest_idx] += 1

            nearest_samples_bulk[client_i] = 0.0
            farthest_samples_bulk[client_i] = 0.0
            assert nearest_samples_bulk.sum() == embedding.size(0)
            assert farthest_samples_bulk.sum() == embedding.size(0)
            nearest_samples_bulk[client_i] = -1e10
            nearest_idx = nearest_samples_bulk.argmax()
            nearest_clients_bulk[nearest_idx] += 1
            farthest_samples_bulk[client_i] = -1e10
            farthest_idx = farthest_samples_bulk.argmax()
            farthest_clients_bulk[farthest_idx] += 1
            customized_peers.append([models[nearest_idx], models[farthest_idx]])

        return customized_peers, embeddings, nearest_clients_bulk, farthest_clients_bulk

    def get_grads_(self, model, server_model):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L487
        """
        grads = []
        for key in server_model.state_dict().keys():
            if "num_batches_tracked" not in key:
                grads.append(
                    model.state_dict()[key].data.clone().detach().flatten()
                    - server_model.state_dict()[key].data.clone().detach().flatten()
                )
        return torch.cat(grads)

    def set_grads_(self, model, server_model, new_grads):
        """
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L494
        """
        start = 0
        for key in server_model.state_dict().keys():
            if "num_batches_tracked" not in key:
                dims = model.state_dict()[key].shape
                end = start + dims.numel()
                model.state_dict()[key].data.copy_(
                    server_model.state_dict()[key].data.clone().detach()
                    + new_grads[start:end].reshape(dims).clone()
                )
                start = end
        return model

    def pcgrad_hierarchy(self, args, client_grads, grad_history=None):
        """
        Projecting conflicting gradients
        Original: https://github.com/CityU-AIM-Group/FedDM/blob/main/main.py#L505
        """
        client_grads_ = torch.stack(client_grads)
        grads = []
        grad_len = grad_history["grad_len"]
        start = 0
        for key in grad_len.keys():
            g_len = grad_len[key]
            end = start + g_len
            layer_grad_history = grad_history[key]
            if layer_grad_history is not None:
                pc_v = layer_grad_history.unsqueeze(0)
                client_grads_layer = client_grads_[:, start:end]
                while True:
                    num = client_grads_layer.size(0)
                    if num > 2:
                        inner_prod = torch.mul(client_grads_layer, pc_v).sum(1)
                        project = inner_prod / (pc_v**2).sum().sqrt()
                        _, ind = project.sort(descending=True)
                        pair_list = []
                        if num % 2 == 0:
                            for i in range(num // 2):
                                pair_list.append([ind[i], ind[num - i - 1]])
                        else:
                            for i in range(num // 2):
                                pair_list.append([ind[i], ind[num - i - 1]])
                            pair_list.append([ind[num // 2]])
                        client_grads_new = []
                        for pair in pair_list:
                            if len(pair) > 1:
                                grad_0 = client_grads_layer[pair[0]]
                                grad_1 = client_grads_layer[pair[1]]
                                inner_prod = torch.dot(grad_0, grad_1)
                                if inner_prod < 0:
                                    # Sustract the conflicting component
                                    grad_pc_0 = (
                                        grad_0 - inner_prod / (grad_1**2).sum() * grad_1
                                    )
                                    grad_pc_1 = (
                                        grad_1 - inner_prod / (grad_0**2).sum() * grad_0
                                    )
                                else:
                                    grad_pc_0 = grad_0
                                    grad_pc_1 = grad_1
                                grad_pc_0_1 = grad_pc_0 + grad_pc_1
                                client_grads_new.append(grad_pc_0_1)
                            else:
                                grad_single = client_grads_layer[pair[0]]
                                client_grads_new.append(grad_single)
                        client_grads_layer = torch.stack(client_grads_new)
                    elif num == 2:
                        grad_pc_0 = client_grads_layer[0]
                        grad_pc_1 = client_grads_layer[1]
                        inner_prod = torch.dot(grad_pc_0, grad_pc_1)
                        if inner_prod < 0:
                            # Sustract the conflicting component
                            grad_pc_0 = (
                                grad_pc_0
                                - inner_prod / (grad_pc_1**2).sum() * grad_pc_1
                            )
                            grad_pc_1 = (
                                grad_pc_1
                                - inner_prod / (grad_pc_0**2).sum() * grad_pc_0
                            )

                        grad_pc_0_1 = grad_pc_0 + grad_pc_1
                        grad_new = grad_pc_0_1 / args.client_num
                        break
                    else:
                        assert False
                gamma = 0.99
                grad_history[key] = gamma * grad_history[key] + (1 - gamma) * grad_new
                grads.append(grad_new)
            else:
                grad_new = client_grads_[:, start:end].mean(0)
                grad_history[key] = grad_new
                grads.append(grad_new)
            start = end
        grad_new = torch.cat(grads)

        return grad_new, grad_history
