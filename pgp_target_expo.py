import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from visdom_logger.logger import VisdomLogger
from custom_models import alexnet
from torch.backends import cudnn


from common_model import Net
from model_adapters import EasyNetAdapter, AlexNetAdapter, ResNetAdapter, VGG16Adapter

cudnn.deterministic = True
import time
import functools
from utils import ParameterType, eval, train, LoggerForSacred

from prune_utils import prune_fc_like, get_prune_index_ratio, create_conv_tensor, \
    get_prune_index_target, L1_criterion, L2_criterion, random_criterion, gng_criterion, init_from_pretrained, \
    in_place_load_state_dict, create_new_bn, get_prune_index_target_with_reset, molchanov_weight_criterion

from custom_models import VGG_CIFAR, LeNet5
import numpy as np

def assert_conv2d_tensor(conv1, conv2):

    # weight check
    assert (conv1.weight.data - conv2.weight.data).sum().item() == 0
    is_bias_1 = conv1.bias is not None
    is_bias_2 = conv2.bias is not None
    assert  is_bias_1 == is_bias_2
    if is_bias_1:
        assert(conv1.bias.data - conv2.bias.data).sum().item() == 0
    conv1_p = list(conv1.parameters())
    conv2_p = list(conv2.parameters())

    for i in range(len(conv1_p)):
        assert (conv1_p[i] - conv2_p[i]).sum().item() == 0

def pgp_target(epochs, target_prune_rate, remove_ratio, test_loader,
        criterion_func, **kwargs):
    optimizer = kwargs["optimizer"]
    model = kwargs["model"]
    cuda = kwargs["cuda"]
    train_loader = kwargs["train_loader"]
    train_ratio = kwargs["train_ratio"]
    initializer_fn = kwargs["initializer_fn"]
    model_adapter = kwargs["model_adapter"]

    logger = kwargs["logger"]
    logger_id = ""
    if "logger_id" in kwargs:
        logger_id = kwargs["logger_id"]


    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True

    final_fn = False
    if "final_fn" in kwargs and kwargs["final_fn"]:
        final_fn = kwargs["final_fn"]
    is_expo = kwargs["is_expo"]


    scheduler = None
    if "scheduler"in kwargs:
        scheduler = kwargs["scheduler"]

    loss_acc = []
    type_list = []
    finished_list = False
    model_architecture = {}
    removed_filters_total = 0
    forced_remove = False
    same_three = 0
    parameters_hard_removed_total = 0


    decay_rates_c = {}
    original_c = {}
    if is_expo:
        for name, parameters in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                decay_rates_c[name] = (np.log(parameters.shape[0]) - np.log(parameters.shape[0] * (1 - target_prune_rate))) / epochs
                original_c[name] = parameters.shape[0]
                model_architecture[name] = []
    else:
        for name, parameters in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
            if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                decay_rates_c[name] = target_prune_rate * parameters.shape[0] / epochs
                original_c[name] = parameters.shape[0]
                model_architecture[name] = []


    for epoch in range(1, epochs + 1):

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
            logger.log_scalar("pgp_target_{}_epoch_time".format(logger_id), start - end, epoch)
            logger.log_scalar("pgp_target_{}_training_loss".format(logger_id), total_loss, epoch)
            logger.log_scalar("pgp_target_{}_before_target_val_acc".format(logger_id), acc, epoch)

        if epoch == epochs:
            remove_ratio = 1

        prune_index_dict, _ = criterion_func(**kwargs)

        out_channels_keep_indexes = []
        in_channels_keep_indexes = []
        reset_indexes = []
        original_out_channels = 0
        first_fc = False
        current_ids = {}
        start_index = None
        last_start_conv = None
        last_keep_index = None
        removed_filters_total_epoch = 0
        reset_filters_total_epoch = 0
        parameters_hard_removed_per_epoch = 0
        parameters_reset_removed = 0

        #print(epoch)
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

                    if is_expo:
                        num_remain_target_prune = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                    else:
                        num_remain_target_prune = original_c[name] - decay_rates_c[name] * epoch #Maybe have to +1 here ?

                    keep_index, reset_index = get_prune_index_target_with_reset(original_out_channels, num_remain_target_prune, remove_ratio,
                                                                    sorted_filters_index, forced_remove)

                    if reset_index is not None:
                        keep_index = torch.cat((keep_index, reset_index))
                    new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                         keep_index, reset_index).to(cuda)
                    model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                    if name not in model_architecture:
                        model_architecture[name] = []
                    model_architecture[name].append(keep_index.shape[0])

                    removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                    if reset_index is not None:
                        reset_filters_total_epoch += len(reset_index)

                    in_c = 3
                    if len(out_channels_keep_indexes) != 0:
                        in_c = out_channels_keep_indexes[-1].shape[0]

                    parameters_hard_removed_per_epoch += (original_out_channels - keep_index.shape[0]) * \
                                                         in_c * parameters.shape[2:].numel()
                    parameters_reset_removed += 0 if reset_index is None or len(reset_index) == 0 else len(
                        reset_index) * in_c * parameters.shape[2:].numel()

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
                    new_fc_tensor = nn.Linear(new_fc_weight.shape[1], new_fc_weight.shape[0], bias=new_fc_bias is not None).to(cuda)
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

                        if is_expo:
                            num_remain_target_prune = original_c[name] * np.exp(-decay_rates_c[name] * (epoch + 1))
                        else:
                            num_remain_target_prune = original_c[name] - decay_rates_c[
                                name] * epoch  # Maybe have to +1 here ?

                        keep_index, reset_index = get_prune_index_target_with_reset(original_out_channels,
                                                                                    num_remain_target_prune,
                                                                                    remove_ratio,
                                                                                    sorted_filters_index, forced_remove)


                        if reset_index is not None:
                            keep_index = torch.cat((keep_index, reset_index))
                        new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                             keep_index, reset_index).to(cuda)
                        model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index,
                                                block_index)

                        if name not in model_architecture:
                            model_architecture[name] = []
                        model_architecture[name].append(keep_index.shape[0])

                        removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                        if reset_index is not None:
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

                            sorted_filters_index = prune_index_dict[d_name]
                            original_out_channels = parameters.shape[0]  # conv_tensor.out_channels
                            last_keep_index, _ = start_index

                            if is_expo:
                                num_remain_target_prune = original_c[d_name] * np.exp(-decay_rates_c[d_name] * (epoch + 1))
                            else:
                                num_remain_target_prune = original_c[d_name] - decay_rates_c[
                                    d_name] * epoch  # Maybe have to +1 here ?

                            keep_index, reset_index = get_prune_index_target_with_reset(original_out_channels,
                                                                                        num_remain_target_prune,
                                                                                        remove_ratio,
                                                                                        sorted_filters_index,
                                                                                        forced_remove)

                            if reset_index is not None:
                                keep_index = torch.cat((keep_index, reset_index))
                            last_start_conv = create_conv_tensor(downsample_cnn, [last_keep_index], initializer_fn,
                                                                 keep_index, reset_index).to(cuda)
                            last_start_conv = [last_start_conv, 0, layer_index, block_index]

                            if d_name not in model_architecture:
                                model_architecture[d_name] = []
                            model_architecture[d_name].append(keep_index.shape[0])

                            removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
                            if reset_index is not None:
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
                        if reset_index is not None:
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
                    model_adapter.set_layer(model, ParameterType.DOWNSAMPLE_WEIGHTS, last_start_conv, tensor_index, layer_index,
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

        new_old_ids = {}
        new_ids = {}
        for k, v in model.named_parameters():
            new_id = id(v)
            new_ids[k] = new_id
            new_old_ids[new_id] = current_ids[k]

        o_state_dict = optimizer.state_dict()
        optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]["lr"], momentum=optimizer.param_groups[0]["momentum"])
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
                index_op_dict[optimizer.param_groups[0]['params'][i]] = (type_list[i], out_channels_keep_indexes[i - 1], None, None)
                first_fc = True
            elif type_list[i] == ParameterType.FC_BIAS:
                continue
            elif type_list[i] == ParameterType.DOWNSAMPLE_BN_B or type_list[i] == ParameterType.DOWNSAMPLE_BN_W or type_list[i] == ParameterType.BN_BIAS or type_list[i] == ParameterType.BN_WEIGHT:
                index_op_dict[optimizer.param_groups[0]['params'][i]] = (
                type_list[i], out_channels_keep_indexes[i], reset_indexes[i], None)
            else:
                index_op_dict[optimizer.param_groups[0]['params'][i]] = (type_list[i], out_channels_keep_indexes[i], reset_indexes[i], in_channels_keep_indexes[i])


        for k,v in index_op_dict.items():
            if v[0] == ParameterType.CNN_WEIGHTS or v[0] == ParameterType.DOWNSAMPLE_WEIGHTS:
                if v[3] is not None and len(v[3]):
                    optimizer.state[k]["momentum_buffer"] = optimizer.state[k]["momentum_buffer"][:, v[3], :, :]
                    if v[2] is not None:
                        optimizer.state[k]["momentum_buffer"][v[2]] = initializer_fn(optimizer.state[k]["momentum_buffer"][v[2]])
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

        removed_filters_total += removed_filters_total_epoch
        parameters_hard_removed_total += parameters_hard_removed_per_epoch
        #("epoch {}: {}".format(epoch, removed_filters_total))


        acc = eval(model, cuda, test_loader,is_break)
        if logger is not None:
            logger.log_scalar("pgp_target_{}_after_target_val_acc".format(logger_id), acc, epoch)
            logger.log_scalar("pgp_target_{}_number of filter removed".format(logger_id), removed_filters_total + reset_filters_total_epoch, epoch)
            logger.log_scalar("pgp_target_{}_acc_number of filter removed".format(logger_id), acc, removed_filters_total + reset_filters_total_epoch)
            logger.log_scalar("pgp_target_{}_acc_number of parameters removed".format(logger_id), acc,
                              parameters_hard_removed_total + parameters_reset_removed)

        loss_acc.append((total_loss / len(train_loader), acc))

    if final_fn[0]:
        optimizer.param_groups[0]["lr"] = final_fn[1]
        #optimizer = optim.SGD(model.parameters(), lr=final_fn[1], momentum=optimizer.param_groups[0]['momentum'])
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
                logger.log_scalar("pgp_{}_epoch_time".format(logger_id), start - end, epoch)
                logger.log_scalar("pgp_{}_training_loss".format(logger_id), total_loss, epoch)
                logger.log_scalar("pgp_{}_before_target_val_acc".format(logger_id), acc, epoch)

    return loss_acc, model_architecture

def prune_one_cnn(block_index, conv_index, conv_tensor, cuda, in_channels_keep_index, initializer_fn, keep_index,
                  layer_index, model, model_adapter, model_architecture, name, param_type, parameters, reset_index):
    new_conv_tensor = create_conv_tensor(conv_tensor, in_channels_keep_index, initializer_fn, keep_index,
                                         name,
                                         parameters, reset_index).to(cuda)
    model_adapter.set_layer(model, param_type, new_conv_tensor, conv_index, layer_index, block_index)
    in_channels_keep_index.append(keep_index.sort()[0])
    if name not in model_architecture:
        model_architecture[name] = []
    model_architecture[name].append(keep_index.shape[0])


def print_parameters(model):

    i = 0
    for name, parameters in model.named_parameters():
        print(str(i) + ":" + name)
        i += 1

def main():

    cuda = torch.device("cuda")
    batch_size = 64
    test_batch_size = 64
    lr = 0.01
    momentum = 0.9

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, num_workers=1)


    model = LeNet5().to(cuda)
    #model = models.resnet18().to(cuda)

    epochs = 5
    prune_rate = 0.5
    remove_ratio = 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    zero_initializer = functools.partial(torch.nn.init.constant_, val=0)

    #logger = VisdomLogger(port=10999)
    #logger = LoggerForSacred(logger)
    logger = None
    pgp_target(epochs, prune_rate, remove_ratio, testloader, gng_criterion,
               cuda=cuda, model=model, train_loader=trainloader, train_ratio=1, prune_once=False,
               initializer_fn=zero_initializer, optimizer=optimizer, logger=logger, model_adapter=AlexNetAdapter(),
               is_expo=True, is_break=True, final_fn=[20, 0.01])

if __name__ == "__main__":

    main()
