import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from prune_utils import prune_fc_like, get_prune_index_ratio, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, \
    gng_criterion, init_from_pretrained, create_conv_tensor, create_new_bn, in_place_load_state_dict

from visdom_logger import VisdomLogger
from common_model import Net
import functools

import torchvision.models as models
from utils import ParameterType, eval, train, LoggerForSacred
from custom_models import alexnet, LeNet5
import torchvision
from model_adapters import EasyNetAdapter, AlexNetAdapter, ResNetAdapter
import pgp
import pickle
import os
import copy


def prune_once(epochs_fn, optimizer, target_prune_remains, test_loader,
               criterion_func, **kwargs):

    model = kwargs["model"]
    cuda = kwargs["cuda"]
    train_loader = kwargs["train_loader"]
    train_ratio = kwargs["train_ratio"]
    model_adapter = kwargs["model_adapter"]
    is_break = kwargs["is_break"]
    loss_acc = []
    type_list = []
    finished_list = False
    model_architecture = {}

    logger = kwargs["logger"]

    if not "logger_id" in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    before_epochs = 0
    prune_index_dict, _ = criterion_func(**kwargs)
    out_channels_keep_indexes = []
    in_channels_keep_indexes = []

    original_out_channels = 0
    first_fc = False
    current_ids = {}
    start_index = None
    last_start_conv = None
    last_keep_index = None
    removed_filters_total_epoch = 0
    removed_parameters_total = 0


    for name, parameters in model.named_parameters():

        current_ids[name] = id(parameters)
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if not finished_list:
            type_list.append(param_type)
        if layer_index == -1:
            # Handling CNN and BN before Resnet
            if param_type == ParameterType.CNN_WEIGHTS:
                sorted_filters_index = prune_index_dict[name]
                conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                original_out_channels = parameters.shape[0]  # conv_tensor.out_channels

                keep_index, reset_index = get_prune_index_target(original_out_channels, target_prune_remains[name],
                                                                 sorted_filters_index)

                new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                     keep_index, reset_index).to(cuda)
                model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                if name not in model_architecture:
                    model_architecture[name] = []
                model_architecture[name].append(keep_index.shape[0])
                removed_filters_total_epoch += original_out_channels - keep_index.shape[0]

                in_c = 3
                if len(out_channels_keep_indexes) != 0:
                    in_c = out_channels_keep_indexes[-1].shape[0]

                removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                            in_c * parameters.shape[2:].numel()
                start_index = (keep_index.sort()[0], reset_index)

                if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                else:
                    in_channels_keep_indexes.append(None)
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.CNN_BIAS:
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(out_channels_keep_indexes[-1])

            elif param_type == ParameterType.BN_WEIGHT:
                bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                keep_index = out_channels_keep_indexes[-1]


                n_bn = create_new_bn(bn_tensor, keep_index, None)
                model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)

                if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_BIAS:

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
                    sorted_filters_index = prune_index_dict[name]
                    conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                    keep_index, reset_index = get_prune_index_target(original_out_channels, target_prune_remains[name],
                                                                     sorted_filters_index)


                    new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                         keep_index, reset_index).to(cuda)
                    model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                            block_index)

                    if name not in model_architecture:
                        model_architecture[name] = []
                    model_architecture[name].append(keep_index.shape[0])
                    removed_filters_total_epoch += original_out_channels - keep_index.shape[0]

                    in_c = conv_tensor.in_channels
                    if len(out_channels_keep_indexes) != 0:
                        in_c = out_channels_keep_indexes[-1].shape[0]

                    removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                in_c * parameters.shape[2:].numel()

                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif tensor_index == 2:

                    downsample_cnn, d_name = model_adapter.get_downsample(model, layer_index, block_index)
                    if downsample_cnn is not None:

                        sorted_filters_index = prune_index_dict[d_name]
                        original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                        last_keep_index, _ = start_index

                        keep_index, reset_index = get_prune_index_target(original_out_channels, target_prune_remains[d_name],
                                                                         sorted_filters_index)

                        last_start_conv = create_conv_tensor(downsample_cnn, [last_keep_index], None,
                                                             keep_index, reset_index).to(cuda)
                        last_start_conv = [last_start_conv, 0, layer_index, block_index]

                        if d_name not in model_architecture:
                            model_architecture[d_name] = []
                        model_architecture[d_name].append(keep_index.shape[0])
                        removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                        removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                    last_keep_index.shape[0] * parameters.shape[2:].numel()
                        start_index = (keep_index.sort()[0], reset_index)

                    conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    keep_index, reset_index = start_index

                    new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                         keep_index, reset_index).to(cuda)
                    model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                            block_index)


                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])

                    removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                    removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                out_channels_keep_indexes[-1].shape[0] * parameters.shape[2:].numel()
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

                in_channels_keep_indexes.append(last_keep_index.sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_WEIGHT:
                bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                keep_index = out_channels_keep_indexes[-1]


                n_bn = create_new_bn(bn_tensor, keep_index, reset_index)
                model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)

                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.BN_BIAS:
                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.DOWNSAMPLE_BN_W:

                bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                keep_index, reset_index = start_index

                n_bn = create_new_bn(bn_tensor, keep_index, reset_index)
                model_adapter.set_layer(model, param_type, n_bn, tensor_index, layer_index, block_index)

                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])

            elif param_type == ParameterType.DOWNSAMPLE_BN_B:
                keep_index, reset_index = start_index

                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(keep_index.sort()[0])


            elif param_type == ParameterType.CNN_BIAS:

                in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                out_channels_keep_indexes.append(out_channels_keep_indexes[-1])
    removed_filters_total = removed_filters_total_epoch



    optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]['lr'], momentum=optimizer.param_groups[0]['momentum'])
    for epoch in range(before_epochs + 1, epochs_fn + 1):
        model.train()

        # One epoch step gradient for target
        optimizer.zero_grad()
        start = time.clock()
        total_loss = train(model, optimizer, cuda, train_loader,is_break)
        acc = eval(model, cuda, test_loader,is_break)

        if logger is not None:
            logger.log_scalar("prune_once_{}_after_target_val_acc".format(logger_id), acc, epoch)
            logger.log_scalar("prune_once_{}_number of filter removed", removed_filters_total, epoch)
            logger.log_scalar("prune_once_{}_acc_number of filter removed".format(logger_id), acc, removed_filters_total)
            logger.log_scalar("prune_once_{}_acc_number of parameters removed".format(logger_id), acc,
                              removed_parameters_total)

        loss_acc.append((total_loss / len(train_loader), acc))

    return loss_acc, model_architecture

if __name__ == "__main__":
    cuda = torch.device("cuda")

    lr = 0.01
    momentum = 0.9

    batch_size = 64
    test_batch_size = 64

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

    model = LeNet5()

    epochs = 170
    prune_rate = 0.05
    remove_ratio = 0.5

    zero_initializer = functools.partial(torch.nn.init.constant_, val=0)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)

    model.to(cuda)
    optimizer_b = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 20 + 1):
        model.train()
        train(model, optimizer_b, cuda, trainloader, True)

    model_adapter = AlexNetAdapter()

    target = {} #np.zeros(len(list(model_architecture.keys())))
    key_index = 0

    for key in [0.9]:
        for k, v in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(k)
            if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                #conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                target[k] = int(v.shape[0] * (1. - key))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_acc, model_architecture = prune_once(epochs, optimizer, target, testloader, L1_criterion, logger=logger,
                                        cuda=cuda, model=model, train_loader=trainloader,
                                        train_ratio=1, initializer_fn=zero_initializer,
                                        model_adapter=AlexNetAdapter(), is_break=True)
