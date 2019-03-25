import sys

sys.path.insert(0, "./frcnn")

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import numpy as np
from model.roi_layers import nms

from torch.utils.data.sampler import Sampler
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time

from utils import ParameterType, LoggerForSacred
from frcnn_utils import FasterRCNN_prepare, train_frcnn, eval_frcnn
from prune_utils import prune_fc_like, get_prune_index_ratio, create_conv_tensor, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, gng_criterion, init_from_pretrained, \
    in_place_load_state_dict, create_new_bn, get_prune_index_target_with_reset, molchanov_weight_criterion, molchanov_weight_criterion_frcnn
from visdom_logger.logger import VisdomLogger

from model_adapters import VGGRCNNAdapter
import functools
from collections import OrderedDict

def pgp_fasterRCNN(epochs, target_prune_rate, remove_ratio,
        criterion_func, **kwargs):

    frcnn_extra = kwargs["frcnn_extra"]
    SHIFT_OPTI = 8
    FREEZE_FIRST_NUM = 10
    if frcnn_extra.net == "res101":
        SHIFT_OPTI = 8



    optimizer = kwargs["optimizer"]
    model = kwargs["model"]
    cuda = kwargs["cuda"]
    initializer_fn = kwargs["initializer_fn"]
    model_adapter = kwargs["model_adapter"]

    logger = kwargs["logger"]
    logger_id = ""
    if "logger_id" in kwargs:
        logger_id = kwargs["logger_id"]


    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True


    till_break = False
    is_conservative = False
    if "is_conservative" in kwargs and kwargs["is_conservative"] is not None:
        is_conservative = kwargs["is_conservative"]
        till_break = True

    kwargs["train_loader"] = frcnn_extra.dataloader_train


    loss_acc = []
    type_list = []
    finished_list = False
    model_architecture = OrderedDict()
    removed_filters_total = 0
    forced_remove = False
    same_three = 0
    parameters_hard_removed_total = 0
    get_weak_fn = get_prune_index_target_with_reset
    lr = optimizer.param_groups[0]['lr']

    decay_rates_c = OrderedDict()
    original_c = OrderedDict()
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            decay_rates_c[name] = target_prune_rate * parameters.shape[0] / epochs
            original_c[name] = parameters.shape[0]
            model_architecture[name] = []


    for epoch in range(1, epochs + 1):

        start = time.clock()
        total_loss = train_frcnn(frcnn_extra, cuda, model, optimizer, is_break)
        end = time.clock()
        if logger is not None:
            logger.log_scalar("pgp_target_frcnn_{}_epoch_time".format(logger_id), time.clock() - end, epoch)
            logger.log_scalar("pgp_target_frcnn_{}_training_loss".format(logger_id), total_loss, epoch)

        if epoch % (frcnn_extra.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, frcnn_extra.lr_decay_gamma)
            lr *= frcnn_extra.lr_decay_gamma

        prune_index_dict, _ = criterion_func(**kwargs)

        out_channels_keep_indexes = []
        in_channels_keep_indexes = []
        reset_indexes = []
        original_out_channels = 0
        first_fc = False
        current_ids = OrderedDict()
        start_index = None
        last_start_conv = None
        last_keep_index = None
        removed_filters_total_epoch = 0
        reset_filters_total_epoch = 0
        parameters_hard_removed_per_epoch = 0
        parameters_reset_removed = 0

        # print(epoch)
        o_state_dict = optimizer.state_dict()
        for name, parameters in model.named_parameters():
            current_ids[name] = id(parameters)
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if not finished_list and parameters.requires_grad:
                type_list.append(param_type)

            if not parameters.requires_grad:
                continue

            if param_type is None:
                reset_indexes.append([])
                out_channels_keep_indexes.append([])
                in_channels_keep_indexes.append([])
                continue

            if layer_index == -1:
                # Handling CNN and BN before Resnet

                if tensor_index == model_adapter.last_layer_index:
                    if param_type == ParameterType.CNN_WEIGHTS:
                        original_out_channels = parameters.shape[0]
                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                        keep_index  = torch.arange(0, original_out_channels).long()
                        reset_index = []

                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                             keep_index, None).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                        in_c = parameters.shape[1]
                        if len(out_channels_keep_indexes) != 0:
                            in_c = out_channels_keep_indexes[-1].shape[0]

                        parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                             in_c * parameters.shape[2:].numel()
                        parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                            reset_index) * in_c * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        else:
                            in_channels_keep_indexes.append(None)
                        out_channels_keep_indexes.append(keep_index.sort()[0])
                    elif param_type == ParameterType.CNN_BIAS:
                        reset_indexes.append(reset_indexes[-1])
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        out_channels_keep_indexes.append(out_channels_keep_indexes[-1])
                    continue

                if param_type == ParameterType.CNN_WEIGHTS:
                    original_out_channels = parameters.shape[0]
                    conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    if name in prune_index_dict:
                        sorted_filters_index = prune_index_dict[name]
                        keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                              sorted_filters_index, forced_remove,
                                                              original_c=original_c[name],
                                                              decay_rates_c=decay_rates_c[name], epoch=epoch)
                        if reset_index is not None:
                            keep_index = torch.cat((keep_index, reset_index))
                    else:
                        keep_index = torch.arange(0, original_out_channels).long()
                        reset_index = []

                    new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                         keep_index, reset_index).to(cuda)
                    model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                    if name not in model_architecture:
                        model_architecture[name] = []
                    model_architecture[name].append(keep_index.shape[0])

                    removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                    reset_filters_total_epoch += len(reset_index)

                    in_c = 3
                    if len(out_channels_keep_indexes) != 0 and len(out_channels_keep_indexes[-1]):
                        in_c = out_channels_keep_indexes[-1].shape[0]

                    parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                         in_c * parameters.shape[2:].numel()
                    parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                        reset_index) * in_c * parameters.shape[2:].numel()

                    start_index = (keep_index.sort()[0], reset_index)

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0 and len(out_channels_keep_indexes[-1]):
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    else:
                        in_channels_keep_indexes.append(None)
                    out_channels_keep_indexes.append(keep_index.sort()[0])
                elif param_type == ParameterType.CNN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    n_bn = create_new_bn(bn_tensor, keep_index, reset_index)
                    model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)
                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.FC_WEIGHTS and first_fc == False:
                    fc_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    new_fc_weight = prune_fc_like(fc_tensor.weight.data, out_channels_keep_indexes[-1],
                                                  original_out_channels)

                    new_fc_bias = None
                    if fc_tensor.bias is not None:
                        new_fc_bias = fc_tensor.bias.data
                    new_fc_tensor = nn.Linear(new_fc_weight.shape[1], new_fc_weight.shape[0],
                                              bias=new_fc_bias is not None).to(cuda)
                    new_fc_tensor.weight.data = new_fc_weight
                    if fc_tensor.bias is not None:
                        new_fc_tensor.bias.data = new_fc_bias
                    model_adapter.set_layer(model, param_type, new_fc_tensor, tensor_index, layer_index, block_index)
                    first_fc = True
                    finished_list = True

            else:

                if param_type == ParameterType.CNN_WEIGHTS:

                    if tensor_index == 1:
                        original_out_channels = parameters.shape[0]
                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                        if name in prune_index_dict:
                            sorted_filters_index = prune_index_dict[name]
                            keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                                  sorted_filters_index, forced_remove,
                                                                  original_c=original_c[name],
                                                                  decay_rates_c=decay_rates_c[name], epoch=epoch)
                            if reset_index is not None:
                                keep_index = torch.cat((keep_index, reset_index))
                        else:
                            keep_index = torch.arange(0, original_out_channels).long()
                            reset_index = []

                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                             keep_index, reset_index).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                                block_index)

                        if name not in model_architecture:
                            model_architecture[name] = []
                        model_architecture[name].append(keep_index.shape[0])

                        removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                        reset_filters_total_epoch += len(reset_index)

                        in_c = conv_tensor.in_channels
                        if len(out_channels_keep_indexes) != 0:
                            in_c = out_channels_keep_indexes[-1].shape[0]

                        parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                             in_c * parameters.shape[
                                                                    2:].numel()
                        parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                            reset_index) * in_c * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        out_channels_keep_indexes.append(keep_index.sort()[0])

                    elif tensor_index == 2:

                        downsample_cnn, d_name = model_adapter.get_downsample(model, layer_index, block_index)
                        if downsample_cnn is not None:
                            original_out_channels = parameters.shape[0]
                            last_keep_index, _ = start_index
                            if d_name in prune_index_dict:
                                sorted_filters_index = prune_index_dict[d_name]
                              # conv_tensor.out_channels


                                keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                                      sorted_filters_index, forced_remove,
                                                                      original_c=original_c[d_name],
                                                                      decay_rates_c=decay_rates_c[d_name], epoch=epoch)

                                if reset_index is not None:
                                    keep_index = torch.cat((keep_index, reset_index))
                            else:
                                keep_index = torch.arange(0, original_out_channels).long()
                                reset_index = []

                            last_start_conv = create_conv_tensor(downsample_cnn, [last_keep_index], initializer_fn,
                                                                 keep_index, reset_index).to(cuda)
                            last_start_conv = [last_start_conv, 0, layer_index, block_index]

                            if d_name not in model_architecture:
                                model_architecture[d_name] = []
                            model_architecture[d_name].append(keep_index.shape[0])

                            removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                            reset_filters_total_epoch += len(reset_index)
                            parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                                 last_keep_index.shape[0] * parameters.shape[2:].numel()
                            parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                                reset_index) * last_keep_index.shape[0] * parameters.shape[2:].numel()
                            start_index = (keep_index.sort()[0], reset_index)

                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                        keep_index, reset_index = start_index

                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                             keep_index, reset_index).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                                block_index)

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])

                        removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                        reset_filters_total_epoch += len(reset_index)
                        parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                             out_channels_keep_indexes[-1].shape[0] * parameters.shape[
                                                                                                      2:].numel()
                        parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                            reset_index) * out_channels_keep_indexes[-1].shape[0] * parameters.shape[2:].numel()

                        out_channels_keep_indexes.append(keep_index.sort()[0])
                        if name not in model_architecture:
                            model_architecture[name] = []
                        model_architecture[name].append(keep_index.shape[0])

                elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS:

                    last_start_conv, tensor_index, layer_index, block_index = last_start_conv
                    model_adapter.set_layer(model, ParameterType.DOWNSAMPLE_WEIGHTS, last_start_conv, tensor_index,
                                            layer_index,
                                            block_index)

                    keep_index, reset_index = start_index
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(last_keep_index.sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    n_bn = create_new_bn(bn_tensor, keep_index, reset_index)
                    model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.DOWNSAMPLE_BN_W:

                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index, reset_index = start_index

                    n_bn = create_new_bn(bn_tensor, keep_index, reset_index)
                    model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.DOWNSAMPLE_BN_B:
                    keep_index, reset_index = start_index
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])


                elif param_type == ParameterType.CNN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

        finished_list = True
        new_old_ids = OrderedDict()
        new_ids = OrderedDict()
        for k, v in model.named_parameters():
            if v.requires_grad:
                new_id = id(v)
                new_ids[k] = new_id
                new_old_ids[new_id] = current_ids[k]

        for layer in range(10):
            for p in model.RCNN_base[layer].parameters(): p.requires_grad = False

        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        optimizer = optim.SGD(params, lr=optimizer.param_groups[0]["lr"],
                              momentum=optimizer.param_groups[0]["momentum"])
        n_new_state_dict = optimizer.state_dict()

        for i, k in enumerate(n_new_state_dict["param_groups"]):

            old_id = new_old_ids[k['params'][0]]
            old_momentum = o_state_dict["state"][old_id]
            n_new_state_dict["state"][k['params'][0]] = old_momentum
        in_place_load_state_dict(optimizer, n_new_state_dict)

        index_op_dict = OrderedDict()
        first_fc = False
        #type_list = [x for x in type_list if x is not None]
        for i in range(len(type_list)):
            if type_list[i] == ParameterType.FC_WEIGHTS and first_fc == False:
                index_op_dict[optimizer.param_groups[i]['params'][0]] = (
                type_list[i], out_channels_keep_indexes[i - 1], None, None)
                first_fc = True
            elif type_list[i] == ParameterType.FC_BIAS:
                continue
            elif type_list[i] == ParameterType.DOWNSAMPLE_BN_B or type_list[i] == ParameterType.DOWNSAMPLE_BN_W or \
                    type_list[i] == ParameterType.BN_BIAS or type_list[i] == ParameterType.BN_WEIGHT:
                index_op_dict[optimizer.param_groups[i]['params'][0]] = (
                    type_list[i], out_channels_keep_indexes[i], reset_indexes[i], None)
            elif type_list[i] is None:
                continue
            elif type_list[i] == ParameterType.CNN_WEIGHTS or type_list[i] == ParameterType.DOWNSAMPLE_WEIGHTS or type_list[i] == ParameterType.CNN_BIAS or type_list == ParameterType.DOWNSAMPLE_BIAS:
                index_op_dict[optimizer.param_groups[i]['params'][0]] = (
                type_list[i], out_channels_keep_indexes[i], reset_indexes[i], in_channels_keep_indexes[i])

        j = 0
        for k, v in index_op_dict.items():

            if v[0] == ParameterType.CNN_WEIGHTS or v[0] == ParameterType.DOWNSAMPLE_WEIGHTS:
                if v[3] is not None and len(v[3]):
                    optimizer.state[k]["momentum_buffer"] = optimizer.state[k]["momentum_buffer"][:, v[3], :, :]
                    if v[2] is not None:
                        optimizer.state[k]["momentum_buffer"][v[2]] = initializer_fn(
                            optimizer.state[k]["momentum_buffer"][v[2]])
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'][v[1], :, :, :]

            elif v[0] == ParameterType.CNN_BIAS or v[0] == ParameterType.BN_WEIGHT or v[0] == ParameterType.BN_BIAS \
                    or v[0] == ParameterType.DOWNSAMPLE_BN_W or v[0] == ParameterType.DOWNSAMPLE_BN_B:
                if v[2] is not None:
                    optimizer.state[k]["momentum_buffer"][v[2]] = initializer_fn(
                        optimizer.state[k]["momentum_buffer"][v[2]])
                optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'][v[1]]
            else:
                optimizer.state[k]['momentum_buffer'] = \
                    prune_fc_like(optimizer.state[k]['momentum_buffer'], v[1], original_out_channels)
            j+= 1
        removed_filters_total += removed_filters_total_epoch
        parameters_hard_removed_total += parameters_hard_removed_per_epoch

        map = eval_frcnn(frcnn_extra, cuda, model, is_break)
        if logger is not None:
            logger.log_scalar("pgp_target_frcnn_{}_after_target_val_acc".format(logger_id), map, epoch)
            logger.log_scalar("pgp_target_frcnn_{}_number of filter removed".format(logger_id), removed_filters_total + reset_filters_total_epoch, epoch)
            logger.log_scalar("pgp_target_frcnn_{}_acc_number of filter removed".format(logger_id), map, removed_filters_total + reset_filters_total_epoch)
            logger.log_scalar("pgp_target_frcnn_{}_acc_number of parameters removed".format(logger_id), map,
                              parameters_hard_removed_total + parameters_reset_removed)
        torch.cuda.empty_cache()

    return loss_acc, model_architecture

def main():

    cuda = torch.device("cuda")
    batch_size = 1
    lr = 0.1
    momentum = 0.9
    frcnn_extra = FasterRCNN_prepare("vgg16", batch_size)
    frcnn_extra.forward()

    fasterRCNN = vgg16(frcnn_extra.imdb_train.classes, pretrained=False, class_agnostic=frcnn_extra.class_agnostic)

    fasterRCNN.create_architecture()
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=momentum, lr=lr)
    fasterRCNN = fasterRCNN.to(cuda)
    zero_initializer = functools.partial(torch.nn.init.constant_, val=0)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)
    epochs = 10
    target_prune_rate = 0.5
    remove_ratio = 0.5


    pgp_fasterRCNN(epochs, target_prune_rate, remove_ratio, molchanov_weight_criterion_frcnn,
        cuda=cuda, model=fasterRCNN, initializer_fn=zero_initializer, optimizer=optimizer,
                   logger=logger, model_adapter=VGGRCNNAdapter(), frcnn_extra=frcnn_extra, is_break=True)



if __name__ == "__main__":
    main()
