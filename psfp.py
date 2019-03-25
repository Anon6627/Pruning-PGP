import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from prune_utils import prune_fc_like, get_prune_index_ratio, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, \
    gng_criterion, init_from_pretrained, create_conv_tensor, create_new_bn

from visdom_logger import VisdomLogger
from common_model import Net
import functools

import torchvision.models as models
from utils import ParameterType, eval, train, LoggerForSacred
from custom_models import alexnet
import torchvision
from model_adapters import EasyNetAdapter, AlexNetAdapter, ResNetAdapter
import pgp
import pickle
import os
from thop import profile

def psfp(epochs, optimizer, target_prune, test_loader,
        criterion_func, **kwargs):

    model = kwargs["model"]
    cuda = kwargs["cuda"]
    train_loader = kwargs["train_loader"]
    train_ratio = kwargs["train_ratio"]
    initializer_fn = kwargs["initializer_fn"]
    logger = kwargs["logger"]
    if not "logger_id" in kwargs:
        logger_id = ""
    else:
        logger_id = kwargs["logger_id"]

    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True

    final_fn = 0
    if "final_fn" in kwargs and kwargs["final_fn"]:
        final_fn = kwargs["final_fn"]

    scheduler = None
    if "scheduler" in kwargs:
        scheduler = kwargs["scheduler"]

    model_adapter = kwargs["model_adapter"]
    loss_acc = []
    model_architecture = {}

    num_train_batch = len(train_loader) * train_ratio

    hoel_magic_value = 0.147 #D always at 1/8
    k = -np.log(hoel_magic_value) / epochs
    a = target_prune / (np.exp(-k * epochs) - 1)

    def num_remain_from_expo(original_c, k, a, epoch):
        decay = a * np.exp(-k * epoch) - a
        num_weak = decay * original_c
        num_remain = original_c - num_weak
        return num_remain

    decay_rates_c = {}
    original_c = {}

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            #decay_rates_c[name] = (np.log(parameters.shape[0]) - np.log(target_prune[name])) / epochs
            original_c[name] = parameters.shape[0]
            model_architecture[name] = []

    finished_list = False
    type_list = []
    forced_remove = False
    removed_filters_total = 0
    removed_parameters_total = 0
    is_last = False

    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        model.train()

        # One epoch step gradient for target
        optimizer.zero_grad()
        start = time.clock()
        total_loss = train(model, optimizer, cuda, train_loader, is_break)
        end = time.clock()
        acc = eval(model, cuda, test_loader, is_break)
        if logger is not None:
            logger.log_scalar("psfp_{}_epoch_time".format(logger_id), time.clock() - end, epoch)
            logger.log_scalar("psfp_{}_training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("psfp_{}_before_target_val_acc".format(logger_id), acc, epoch)
        optimizer.zero_grad()

        prune_index_dict, _ = criterion_func(**kwargs)
        out_channels_keep_indexes = []
        in_channels_keep_indexes = []
        reset_indexes = []
        first_fc = False
        removed_filters_total_epoch = 0


        for name, parameters in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if not finished_list:
                type_list.append(param_type)

            if layer_index == -1:
                # Handling CNN and BN before Resnet
                if param_type == ParameterType.CNN_WEIGHTS:
                    sorted_filters_index = prune_index_dict[name]
                    conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                    #prune_target = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                    prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                    keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                     sorted_filters_index, forced_remove)

                    if len(reset_indexes) != 0:
                        conv_tensor.weight.data[:, reset_indexes[-1], :, :] = initializer_fn(conv_tensor.weight.data[:, reset_indexes[-1], :, :])


                    conv_tensor.weight.data[reset_index] = initializer_fn(conv_tensor.weight.data[reset_index])
                    if conv_tensor.bias is not None:
                        conv_tensor.bias.data[reset_index] = initializer_fn(conv_tensor.bias.data[reset_index])
                    model_architecture[name].append(keep_index.shape[0])
                    removed_filters_total_epoch += reset_index.shape[0]

                    in_c = 3
                    if len(reset_indexes) != 0:
                        in_c =  reset_indexes[-1].shape[0]
                    removed_parameters_total =+ reset_index.shape[0] * in_c * parameters.shape[2:].numel()

                    start_index = (keep_index.sort()[0], reset_index)
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

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    bn_tensor.running_var.data[reset_index] = initializer_fn(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = initializer_fn(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = initializer_fn(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = initializer_fn(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.FC_WEIGHTS and first_fc == False:
                    fc_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                    reset_index = reset_indexes[-1]

                    reshaped_fc = fc_tensor.weight.data.view(fc_tensor.weight.data.shape[0], original_out_channels, -1)
                    reshaped_fc[:, reset_index, :] = initializer_fn(reshaped_fc[:, reset_index, :])
                    fc_tensor.weight.data = reshaped_fc.view(fc_tensor.weight.data.shape[0], -1)

                    first_fc = True
                    finished_list = True

            else:
                if param_type == ParameterType.CNN_WEIGHTS:

                    if tensor_index == 1:
                        sorted_filters_index = prune_index_dict[name]
                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                        original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                        prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                        keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                         sorted_filters_index, forced_remove)

                        if len(reset_indexes) != 0:
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :] = initializer_fn(
                                conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                        conv_tensor.weight.data[reset_index] = initializer_fn(conv_tensor.weight.data[reset_index])
                        if conv_tensor.bias is not None:
                            conv_tensor.bias.data[reset_index] = initializer_fn(conv_tensor.bias.data[reset_index])
                        model_architecture[name].append(keep_index.shape[0])

                        in_c = conv_tensor.in_channels
                        if len(out_channels_keep_indexes) != 0:
                            in_c = reset_indexes[-1].shape[0]
                        removed_filters_total_epoch += reset_index.shape[0]
                        removed_parameters_total = + reset_index.shape[0] * in_c * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None and len(out_channels_keep_indexes) != 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        else:
                            in_channels_keep_indexes.append(None)
                        out_channels_keep_indexes.append(keep_index.sort()[0])

                    elif tensor_index == 2:

                        downsample_cnn, d_name = model_adapter.get_downsample(model, layer_index, block_index)
                        if downsample_cnn is not None:

                            sorted_filters_index = prune_index_dict[d_name]
                            original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                            last_keep_index, last_reset_index = start_index
                            prune_target = original_c[name]
                            #prune_target = num_remain_from_expo(original_c[name], k, a, epoch)
                            #prune_target = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                            keep_index, reset_index = get_prune_index_target(original_out_channels, prune_target,
                                                                             sorted_filters_index, forced_remove)
                            downsample_cnn.weight.data[:, last_reset_index, :, :] = initializer_fn(
                                downsample_cnn.weight.data[:, last_reset_index, :, :])

                            downsample_cnn.weight.data[reset_index] = initializer_fn(downsample_cnn.weight.data[reset_index])
                            if downsample_cnn.bias is not None:
                                downsample_cnn.bias.data[reset_index] = initializer_fn(downsample_cnn.bias.data[reset_index])

                            if d_name not in model_architecture:
                                model_architecture[d_name] = []
                            model_architecture[d_name].append(keep_index.shape[0])


                            removed_filters_total_epoch += reset_index.shape[0]
                            removed_parameters_total = + reset_index.shape[0] * last_reset_index.shape[0] * parameters.shape[2:].numel()
                            start_index = (keep_index.sort()[0], reset_index)

                        conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                        keep_index, reset_index = start_index

                        if len(reset_indexes) != 0:
                            conv_tensor.weight.data[:, reset_indexes[-1], :, :] = initializer_fn(
                                conv_tensor.weight.data[:, reset_indexes[-1], :, :])

                        conv_tensor.weight.data[reset_index] = initializer_fn(conv_tensor.weight.data[reset_index])
                        if conv_tensor.bias is not None:
                            conv_tensor.bias.data[reset_index] = initializer_fn(conv_tensor.bias.data[reset_index])

                        removed_filters_total_epoch += reset_index.shape[0]
                        removed_parameters_total = + reset_index.shape[0] * reset_indexes[-1].shape[
                            0] * parameters.shape[2:].numel()

                        reset_indexes.append(reset_index)
                        if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                            in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                        out_channels_keep_indexes.append(keep_index.sort()[0])
                        model_architecture[name].append(keep_index.shape[0])


                elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS:

                    keep_index, reset_index = start_index
                    reset_indexes.append(reset_index)
                    in_channels_keep_indexes.append(last_keep_index.sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_WEIGHT:
                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index = out_channels_keep_indexes[-1]
                    reset_index = reset_indexes[-1]

                    bn_tensor.running_var.data[reset_index] = initializer_fn(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = initializer_fn(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = initializer_fn(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = initializer_fn(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) == 0:
                        in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.BN_BIAS:
                    reset_indexes.append(reset_indexes[-1])
                    in_channels_keep_indexes.append(out_channels_keep_indexes[-1].sort()[0])
                    out_channels_keep_indexes.append(keep_index.sort()[0])

                elif param_type == ParameterType.DOWNSAMPLE_BN_W:

                    bn_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

                    keep_index, reset_index = start_index

                    bn_tensor.running_var.data[reset_index] = initializer_fn(bn_tensor.running_var.data[reset_index])
                    bn_tensor.running_mean.data[reset_index] = initializer_fn(bn_tensor.running_mean.data[reset_index])
                    bn_tensor.weight.data[reset_index] = initializer_fn(bn_tensor.weight.data[reset_index])
                    bn_tensor.bias.data[reset_index] = initializer_fn(bn_tensor.bias.data[reset_index])

                    reset_indexes.append(reset_index)
                    if out_channels_keep_indexes is not None or len(out_channels_keep_indexes) != 0:
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

        #print("Total: {}".format(removed_filters_total))
        removed_filters_total = removed_filters_total_epoch
        if removed_filters_total - removed_filters_total_epoch == 0:
            forced_remove = True
        acc = eval(model, cuda, test_loader,is_break)
        if logger is not None:
            logger.log_scalar("psfp_{}_after_target_val_acc".format(logger_id), acc, epoch)
            logger.log_scalar("psfp_{}_number of filter removed".format(logger_id), removed_filters_total, epoch)
            logger.log_scalar("psfp_{}_acc_number of filter removed".format(logger_id), acc, removed_filters_total)
            logger.log_scalar("psfp_{}_acc_number of parameters removed".format(logger_id), acc, removed_parameters_total)

        loss_acc.append((total_loss / len(train_loader), acc))

    if final_fn[0]:
        optimizer.param_groups[0]["lr"] = final_fn[1]
        for epoch in range(epochs + 1, epochs + final_fn[0] + 1):
            if scheduler is not None:
                scheduler.step()
            model.train()

            # One epoch step gradient for target
            optimizer.zero_grad()

            start = time.clock()
            total_loss = train(model, optimizer, cuda, train_loader,is_break)
            end = time.clock()
            acc = eval(model, cuda, test_loader,is_break)

            if logger is not None:
                logger.log_scalar("psfp_{}_epoch_time".format(logger_id), start - end, epoch)
                logger.log_scalar("psfp_{}_training_loss".format(logger_id), total_loss, epoch)
                logger.log_scalar("psfp_{}_before_target_val_acc".format(logger_id), acc, epoch)

    flops, params = profile(model, input_size=train_loader.dataset[0][0].unsqueeze(0).shape)
    logger.log_scalar("ipsfp_{}_flops_counts".format(logger_id), flops, 0)
    logger.log_scalar("psfp_{}_params_counts".format(logger_id), params, 0)
    return loss_acc, model_architecture

if __name__ == "__main__":
    cuda = torch.device("cuda")
    batch_size = 128
    test_batch_size = 128
    lr = 0.01
    momentum = 0.9


    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    model_pgp = alexnet().to(cuda)

    epochs = 170
    prune_rate = 0.05
    remove_ratio = 0.5
    optimizer_pgp = optim.SGD(model_pgp.parameters(), lr=lr, momentum=momentum)
    zero_initializer = functools.partial(torch.nn.init.constant_, val=0)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)

    if not os.path.exists("temp_target.p"):


        loss_acc, model_architecture = pgp.pgp(epochs, prune_rate, remove_ratio, testloader, gng_criterion,
                                               cuda=cuda, model=model_pgp, train_loader=trainloader, train_ratio=1, prune_once=False,
                                               initializer_fn=zero_initializer, optimizer=optimizer_pgp, logger=logger, model_adapter=AlexNetAdapter(), is_conservative=False)

        pickle.dump(model_architecture, open("temp_target.p", "wb"))

    model_architecture = pickle.load(open("temp_target.p", "rb"))

    target = {} #np.zeros(len(list(model_architecture.keys())))
    for k, v in model_architecture.items():
        target[k] = v[-1]

    model_psfp = alexnet().to(cuda)

    optimizer_psfp = optim.SGD(model_psfp.parameters(), lr=lr, momentum=momentum)
    loss_acc, model_architecture = psfp(epochs, optimizer_psfp, target, testloader, L2_criterion, logger=logger,
                                           cuda=cuda, model=model_psfp, train_loader=trainloader,
                                           train_ratio=1, initializer_fn=zero_initializer,
                                           model_adapter=AlexNetAdapter())

    prune_rate = .1
    remove_ratio = 1.


