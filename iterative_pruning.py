import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from prune_utils import prune_fc_like, get_prune_index_ratio, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, \
    gng_criterion, init_from_pretrained, create_conv_tensor, create_new_bn, in_place_load_state_dict, molchanov_criterion

from visdom_logger import VisdomLogger
from thop import profile
from common_model import Net
import functools

import torchvision.models as models
from utils import ParameterType, eval, train, LoggerForSacred
from custom_models import alexnet, VGG_CIFAR
import torchvision
from model_adapters import EasyNetAdapter, AlexNetAdapter, ResNetAdapter, VGG16Adapter
import pgp
import pickle
import os
import copy
import collections
from experiments.metrics_from_mongo import  get_original_num_of_filters
import resnet_cifar

def finalize_weakest_list(l, model,  model_adapter):
    prune_name_dict = {}

    for i, e in enumerate(l):
        if not e[0] in prune_name_dict:
            prune_name_dict[e[0]] = [e[0], torch.LongTensor([e[1]]), e[2]]
        else:
            a = torch.LongTensor([e[1]])
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(e[0])
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            keep_i = torch.cat((prune_name_dict[e[0]][1], a))
            if conv_tensor.out_channels > keep_i.shape[0]:
                prune_name_dict[e[0]][1] = keep_i

    return prune_name_dict

def insert_sort_list(l, value,name, index, is_downsample, prune_each_steps):
    #name, index, downsample, value

    if len(l) == 0:
        l.append((name, index, is_downsample, value))
        return l

    for i,e in enumerate(l):
        if value <= e[3]:
            l.insert(i, [name, index, is_downsample, value])
            if len(l) > prune_each_steps:
                l = l[:prune_each_steps]
            return l
    if len(l) < prune_each_steps:
        l.append([name, index, is_downsample, value])
    return l


def iterative_pruning(epochs_fn, target_prune, prune_each_steps, test_loader,
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
    optimizer = kwargs["optimizer"]

    key_pts = np.asarray([0.3, 0.5, 0.7, 0.9])

    ki = 0

    logger = kwargs["logger"]

    if not "logger_id" in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    original_num_filers = get_original_num_of_filters(model, model_adapter)
    target_num_prune = int(original_num_filers * target_prune / prune_each_steps)
    key_pts = key_pts * original_num_filers
    removed_filter_total = 0
    removed_parameters_total = 0

    for num_of_removed in range(1, target_num_prune + 1):



        prune_index_dict, values_indexes = criterion_func(**kwargs)

        out_channels_keep_indexes = []
        in_channels_keep_indexes = []

        original_out_channels = 0
        first_fc = False
        current_ids = {}
        start_index = None
        last_start_conv = None
        last_keep_index = None
        weakest_by_name = {}
        weakest_val = 999999999
        list_name_param = []

        weakest_list = []
        for k, v in values_indexes.items():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(k)
            if len(v.shape) == 0 or v.shape[0] == 1 or (tensor_index == 2 and layer_index != -1):
                continue

            limit = prune_each_steps if v.shape[0] > prune_each_steps else v.shape[0]

            for i in range(v[:limit].shape[0]):
                if param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                    _, conv2_name = model_adapter.get_conv2_from_downsample(model, layer_index, block_index)
                    weakest_list = insert_sort_list(weakest_list, v[i], conv2_name, prune_index_dict[k][i], True, prune_each_steps)
                else:
                    weakest_list = insert_sort_list(weakest_list, v[i], k, prune_index_dict[k][i], False,
                                                    prune_each_steps)

        prune_name_dict = finalize_weakest_list(weakest_list, model,  model_adapter)

        for name, parameters in model.named_parameters():

            current_ids[name] = id(parameters)
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if not finished_list:
                type_list.append(param_type)
                list_name_param.append(name)

            if layer_index == -1:
                # Handling CNN and BN before Resnet
                if param_type == ParameterType.CNN_WEIGHTS:

                    conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    original_out_channels = parameters.shape[0]  # conv_tensor.out_channels

                    reset_index = None
                    if not prune_name_dict is None and name in prune_name_dict:
                        _, filtered_index, _ = prune_name_dict[name]
                        original_index = torch.arange(0, original_out_channels)
                        mask = torch.ones_like(original_index)
                        mask[filtered_index] = 0
                        keep_index = original_index[mask.nonzero()].squeeze()
                        if original_index.shape[0] - filtered_index.shape[0] == 0:
                            keep_index = torch.arange(0, original_out_channels).long()
                        elif original_index.shape[0] - filtered_index.shape[0] == 1:
                            keep_index = torch.LongTensor([keep_index])
                    else:
                        keep_index = torch.arange(0, original_out_channels).long()


                    new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                         keep_index, reset_index).to(cuda)
                    model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                    if name not in model_architecture:
                        model_architecture[name] = []
                    model_architecture[name].append(keep_index.shape[0])
                    start_index = (keep_index.sort()[0], reset_index)

                    in_c = conv_tensor.in_channels
                    if len(out_channels_keep_indexes) != 0:
                        in_c = out_channels_keep_indexes[-1].shape[0]

                    removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                in_c * parameters[2:].numel()
                    removed_filter_total += original_out_channels - keep_index.shape[0]

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
                    del bn_tensor
                    torch.cuda.empty_cache()

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

                    del fc_tensor
                    torch.cuda.empty_cache()


                    first_fc = True
                    finished_list = True

            else:

                if param_type == ParameterType.CNN_WEIGHTS:

                    if tensor_index == 1:
                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                        original_out_channels = parameters.shape[0]  # conv_tensor.out_channels

                        reset_index = None
                        if not prune_name_dict is None and name in prune_name_dict:
                            _, filtered_index, _ = prune_name_dict[name]
                            original_index = torch.arange(0, original_out_channels)
                            mask = torch.ones_like(original_index)
                            mask[filtered_index] = 0
                            keep_index = original_index[mask.nonzero()].squeeze()
                            if original_index.shape[0] - filtered_index.shape[0] == 0:
                                keep_index = torch.arange(0, original_out_channels).long()
                            elif original_index.shape[0] - filtered_index.shape[0] == 1:
                                keep_index = torch.LongTensor([keep_index])

                        else:
                            keep_index = torch.arange(0, original_out_channels).long()


                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                             keep_index, reset_index).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                                block_index)

                        if name not in model_architecture:
                            model_architecture[name] = []
                        model_architecture[name].append(keep_index.shape[0])

                        in_c = conv_tensor.in_channels
                        if len(out_channels_keep_indexes) != 0:
                            in_c = out_channels_keep_indexes[-1].shape[0]
                        removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                    in_c * parameters[2:].numel()
                        removed_filter_total += original_out_channels - keep_index.shape[0]

                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        out_channels_keep_indexes.append(keep_index.sort()[0])


                    elif tensor_index == 2:

                        downsample_cnn, d_name = model_adapter.get_downsample(model, layer_index, block_index)

                        if downsample_cnn is not None:
                            original_out_channels = parameters.shape[0]  # conv_tensor.out_channels

                            last_keep_index, _ = start_index

                            reset_index = None
                            if not prune_name_dict is None and name in prune_name_dict and prune_name_dict[name][2]:
                                _, filtered_index, _ = prune_name_dict[name]
                                original_index = torch.arange(0, original_out_channels)
                                mask = torch.ones_like(original_index)
                                mask[filtered_index] = 0
                                keep_index = original_index[mask.nonzero()].squeeze()
                                if original_index.shape[0] - filtered_index.shape[0] == 0:
                                    keep_index = torch.arange(0, original_out_channels).long()
                                elif original_index.shape[0] - filtered_index.shape[0] == 1:
                                    keep_index = torch.LongTensor([keep_index])
                            else:
                                keep_index = torch.arange(0, original_out_channels).long()


                            last_start_conv = create_conv_tensor(downsample_cnn, [last_keep_index], None,
                                                                 keep_index, reset_index).to(cuda)
                            last_start_conv = [last_start_conv, 0, layer_index, block_index]

                            if d_name not in model_architecture:
                                model_architecture[d_name] = []
                            model_architecture[d_name].append(keep_index.shape[0])
                            start_index = (keep_index.sort()[0], reset_index)
                            removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                        last_keep_index.shape[0] * parameters[2:].numel()
                            removed_filter_total += original_out_channels - keep_index.shape[0]

                        original_out_channels = parameters.shape[0]
                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                        keep_index, reset_index = start_index

                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, None,
                                                             keep_index, reset_index).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                                block_index)

                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])

                        removed_parameters_total += (original_out_channels - keep_index.shape[0]) * \
                                                    out_channels_keep_indexes[-1].shape[0] * parameters[2:].numel()
                        removed_filter_total += original_out_channels - keep_index.shape[0]

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

        if num_of_removed > 1:
            new_old_ids = {}
            new_ids = {}
            for k, v in model.named_parameters():
                new_id = id(v)
                new_ids[k] = new_id
                new_old_ids[new_id] = current_ids[k]

            o_state_dict = optimizer.state_dict()
            optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]["lr"],
                                  momentum=optimizer.param_groups[0]["momentum"])
            n_new_state_dict = optimizer.state_dict()

            for k in n_new_state_dict["param_groups"][0]["params"]:
                old_id = new_old_ids[k]
                old_momentum = o_state_dict["state"][old_id]
                n_new_state_dict["state"][k] = old_momentum
            in_place_load_state_dict(optimizer, n_new_state_dict)

            index_op_dict = {}
            first_fc = False
            for i in range(len(type_list)):
                if type_list[i] == ParameterType.FC_WEIGHTS and first_fc == False:
                    index_op_dict[optimizer.param_groups[0]['params'][i]] = (
                    type_list[i], out_channels_keep_indexes[i - 1], None, None)
                    first_fc = True
                elif type_list[i] == ParameterType.FC_BIAS:
                    continue
                elif type_list[i] == ParameterType.DOWNSAMPLE_BN_B or type_list[i] == ParameterType.DOWNSAMPLE_BN_W or \
                        type_list[i] == ParameterType.BN_BIAS or type_list[i] == ParameterType.BN_WEIGHT:
                    index_op_dict[optimizer.param_groups[0]['params'][i]] = (
                        type_list[i], out_channels_keep_indexes[i], None, None)
                else:
                    index_op_dict[optimizer.param_groups[0]['params'][i]] = (
                    type_list[i], out_channels_keep_indexes[i], None, in_channels_keep_indexes[i])

            for k, v in index_op_dict.items():
                if v[0] == ParameterType.CNN_WEIGHTS or v[0] == ParameterType.DOWNSAMPLE_WEIGHTS:
                    if v[3] is not None:
                        optimizer.state[k]["momentum_buffer"] = optimizer.state[k]["momentum_buffer"][:, v[3], :, :]

                    optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'][v[1], :, :, :]

                elif v[0] == ParameterType.CNN_BIAS or v[0] == ParameterType.BN_WEIGHT or v[0] == ParameterType.BN_BIAS \
                        or v[0] == ParameterType.DOWNSAMPLE_BN_W or v[0] == ParameterType.DOWNSAMPLE_BN_B:

                    optimizer.state[k]['momentum_buffer'] = optimizer.state[k]['momentum_buffer'][v[1]]
                else:
                    optimizer.state[k]['momentum_buffer'] = \
                        prune_fc_like(optimizer.state[k]['momentum_buffer'], v[1], original_out_channels)


        if num_of_removed == 1:
            optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]['lr'], momentum=optimizer.param_groups[0]['momentum'])
            torch.cuda.empty_cache()
        for epoch in range(1, epochs_fn):
            model.train()
            optimizer.zero_grad()
            total_loss = train(model, optimizer, cuda, train_loader,is_break)
        acc = eval(model, cuda, test_loader,is_break)

        if logger is not None:
            logger.log_scalar("iterative_pruning_{}_after_target_val_acc".format(logger_id), acc, num_of_removed)
            logger.log_scalar("iterative_pruning_{}_number of filter removed", removed_filter_total, epoch)
            logger.log_scalar("iterative_pruning_{}_acc_number of filter removed".format(logger_id), acc, removed_filter_total)
            logger.log_scalar("iterative_pruning_{}_acc_parameters_removed".format(logger_id), acc,
                              removed_parameters_total)

            #print("{}:{}:{}".format(removed_filter_total, key_pts[ki], original_num_filers))
            if ki < key_pts.shape[0] and removed_filter_total >= key_pts[ki]:
                flops, params = profile(model, input_size=train_loader.dataset[0][0].unsqueeze(0).shape)
                logger.log_scalar("iterative_pruning_{}_flops_counts".format(logger_id), flops, key_pts[ki])
                logger.log_scalar("iterative_pruning_{}_params_counts".format(logger_id), params, key_pts[ki])
                ki += 1


            loss_acc.append((total_loss / len(train_loader), acc))

        if removed_filter_total > target_prune * original_num_filers:
            print("{}/{}".format(removed_filter_total, original_num_filers))
            break
        #print("{}:{}/{}".format(num_of_removed,removed_total, original_num_filers))

    flops, params = profile(model, input_size=train_loader.dataset[0][0].unsqueeze(0).shape)
    logger.log_scalar("iterative_pruning_{}_flops_counts".format(logger_id), flops, key_pts[-1])
    logger.log_scalar("iterative_pruning_{}_params_counts".format(logger_id), params, key_pts[-1])
    return loss_acc, model_architecture

if __name__ == "__main__":
    cuda = torch.device("cuda")
    batch_size = 128
    test_batch_size = 128
    lr = 0.01
    momentum = 0.9

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)


    model = resnet_cifar.resnet56_cifar().to(cuda)
    #model = VGG_CIFAR().to(cuda)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)
    if not os.path.exists("resnet56_trained_cifar10.p"):
        optimizer_b = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for epoch in range(1, 20 + 1):
            model.train()
            train(model, optimizer_b, cuda, trainloader, True)
        torch.save(model, "resnet56_trained_cifar10.p")
    model = torch.load("resnet56_trained_cifar10.p").to(cuda)


    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_acc, model_architecture = iterative_pruning(2, 0.9, 5, testloader, L1_criterion, cuda=cuda, model=model,
                                                                optimizer=optimizer,
                                                               train_loader=trainloader,
                                                               train_ratio=1,
                                                               initializer_fn=None,
                                                               model_adapter=ResNetAdapter(), is_break=True, logger=logger)