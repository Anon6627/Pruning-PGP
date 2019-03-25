import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._six import container_abcs
import torch.optim as optim
from utils import ParameterType
from collections import defaultdict
from itertools import chain
import copy
import numpy as np

def create_new_bn(bn_tensor, keep_index, reset_index):
    n_running_var_data = bn_tensor.running_var.data
    n_running_mean_data = bn_tensor.running_mean.data
    n_weight_data = bn_tensor.weight.data
    n_bias_data = bn_tensor.bias.data
    if reset_index is not None and len(reset_index) != 0:
        n_running_var_data[reset_index] = 0
        n_running_mean_data[reset_index] = 0
        n_weight_data[reset_index] = 0
        n_bias_data[reset_index] = 0
    n_running_var_data = bn_tensor.running_var.data[keep_index]
    n_running_mean_data = bn_tensor.running_mean.data[keep_index]
    n_weight_data = bn_tensor.weight.data[keep_index]
    n_bias_data = bn_tensor.bias.data[keep_index]
    n_bn = nn.BatchNorm2d(keep_index.shape[0])
    n_bn.running_var.data = n_running_var_data
    n_bn.running_mean.data = n_running_mean_data
    n_bn.weight.data = n_weight_data
    n_bn.bias.data = n_bias_data
    return n_bn


def in_place_load_state_dict(optimizer, state_dict):

    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain(*(g['params'] for g in saved_groups)),
                  chain(*(g['params'] for g in groups)))}

    def cast(param, value):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if param.is_floating_point():
                value = value.to(param.dtype)
            value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        new_group['params'] = group['params']
        return new_group

    param_groups = [
        update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    optimizer.__setstate__({'state': state, 'param_groups': param_groups})



def create_conv_tensor(old_tensor, out_channels_keep_indexes, initializer_fn, keep_index, reset_index):

    new_weights = old_tensor.weight.detach().cpu()
    is_bias = old_tensor.bias is not None

    if len(out_channels_keep_indexes) and len(out_channels_keep_indexes[-1]) != 0:
        new_weights = new_weights[:, out_channels_keep_indexes[-1], :, :]

    if reset_index is not None and len(reset_index) > 0 and initializer_fn is not None:
        new_weights[reset_index] = initializer_fn(new_weights[reset_index])


    new_weights = new_weights[keep_index.sort()[0]]

    new_conv_tensor = nn.Conv2d(new_weights.shape[1],  # Input channels
                                keep_index.shape[0],  # Output channels
                                old_tensor.kernel_size[0], bias=is_bias, stride=old_tensor.stride, padding=old_tensor.padding)

    if is_bias:
        if reset_index is not None and len(reset_index) > 0 and initializer_fn is not None:
            old_tensor.bias.data[reset_index] = initializer_fn(old_tensor.bias.data[reset_index])
        new_conv_tensor.bias.data = old_tensor.bias.data[keep_index.sort()[0]]

    new_conv_tensor.weight.data = new_weights
    new_conv_tensor.training = old_tensor.training



    return new_conv_tensor

def prune_fc_like(fc_like_tensor_data, removed_indexes, original_out_channels):
    reshaped_fc_grad = fc_like_tensor_data.view(fc_like_tensor_data.shape[0], original_out_channels, -1)
    pruned_reshaped_fc = reshaped_fc_grad[:, removed_indexes.sort()[0], :]
    fc_tensor_grad = pruned_reshaped_fc.view(fc_like_tensor_data.shape[0], -1)
    return fc_tensor_grad

def get_prune_index_ratio(current_total_filters, prune_ratio, remove_ratio, sorted_filters_index, forced=False):

    if len(sorted_filters_index.shape) == 0:
        keep_index = torch.LongTensor([0])
        return keep_index, None

    prune_threshold_index = int(current_total_filters * prune_ratio)

    if forced and  prune_threshold_index <= 0 and current_total_filters > 1:
        prune_threshold_index = 1

    reset_remove_index, keep_index = (sorted_filters_index[:prune_threshold_index],
                                      sorted_filters_index[prune_threshold_index:])
    reset_remove_threshold = int(prune_threshold_index * remove_ratio)
    if reset_remove_threshold == 0 and current_total_filters > 1:
        reset_remove_threshold = 1
    _, reset_index = (reset_remove_index[:reset_remove_threshold],
                                 reset_remove_index[reset_remove_threshold:])

    return keep_index, reset_index

def get_prune_index_target_with_reset(current_total_filters, num_remain_target_prune, remove_ratio, sorted_filters_index, forced=False):


    num_of_weaks = int(current_total_filters - num_remain_target_prune)
    if num_of_weaks == current_total_filters:
        num_of_weaks = current_total_filters - 1

    if forced and num_of_weaks <= 0 and current_total_filters > 1:
        num_of_weaks = 1


    if num_of_weaks != 0:
        weak_index, keep_index = (sorted_filters_index[:num_of_weaks],
                                          sorted_filters_index[num_of_weaks:])
        reset_remove_threshold = int(weak_index.shape[0] * remove_ratio)
        _, reset_index = (weak_index[:reset_remove_threshold],
                          weak_index[reset_remove_threshold:])

        if len(reset_index) == 1 and current_total_filters == 1:
            reset_index = None


    else:
        keep_index = torch.arange(0, current_total_filters)
        reset_index = None

    return keep_index, reset_index

def get_prune_index_target(current_total_filters, num_remain_target_prune, sorted_filters_index, forced=False):
    num_of_weaks = int(current_total_filters - num_remain_target_prune)
    if num_of_weaks == current_total_filters:
        num_of_weaks = current_total_filters - 1

    if forced and num_of_weaks <= 0 and current_total_filters > 1:
        num_of_weaks = 1

    reset_index, keep_index = (sorted_filters_index[:num_of_weaks],
                                      sorted_filters_index[num_of_weaks:])

    if len(reset_index) == 1 and current_total_filters == 1:
        reset_index = None

    return keep_index, reset_index

def L1_criterion(**kwargs):
    model = kwargs["model"]
    indexes_dict = {}
    values_indexes = {}
    model_adapter = kwargs["model_adapter"]
    for name, parameters in model.named_parameters():
        param_type, *_ = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS: #Only handling convolution for now
            filters_L1 = parameters.data.view(parameters.shape[0], -1).norm(dim=1, p=1)
            values_indexes[name], indexes_dict[name] = filters_L1.sort()

    return indexes_dict, values_indexes

def L2_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    model_adapter = kwargs["model_adapter"]
    for name, parameters in model.named_parameters():
        param_type, *_ = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS: #Only handling convolution for now
            filters_L1 = parameters.data.view(parameters.shape[0], -1).norm(dim=1, p=2)
            _, indexes_dict[name] = filters_L1.sort()

    return indexes_dict, None

def random_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    model_adapter = kwargs["model_adapter"]
    for name, parameters in model.named_parameters():
        param_type, *_ = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS: #Only handling convolution for now
            indexes_dict[name] = torch.randperm(parameters.data.shape[0])
    return indexes_dict, None


def gng_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cuda), target.to(cuda)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()


    for name, parameters in model.named_parameters():
        param_type, *_ = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS: #Only handling convolution for now
            grad = parameters.grad
            grad_criteria = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1)
            # grad_criteria = grad.view(grad.shape[0], -1).sum(dim=1)
            grad_norm_by_filter = grad_criteria
            _, indexes_dict[name] = torch.sort(grad_norm_by_filter.squeeze(), descending=False)
    return indexes_dict, None

def molchanov_criterion(**kwargs):
    indexes_dict = {}
    values_indexes = {}
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    optimizer = kwargs["optimizer"]
    handle = []

    model_c = model
    optimizer_c = torch.optim.SGD(model_c.parameters(), lr=optimizer.param_groups[0]["lr"],
                                  momentum=optimizer.param_groups[0]["momentum"])


    all_fm = {}
    def forward_hook(self, input, output):
        nonlocal all_fm
        all_fm[self] = F.relu(output.detach().cpu())

    all_rank = {}
    def backward_hook(self, grad_input, grad_output):
        nonlocal all_rank

        rank = torch.sum(all_fm[self] * grad_output[0].detach().cpu(), dim=0, keepdim=True)
        rank = rank.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        rank = rank / (all_fm[self].shape[0] * all_fm[self].shape[2] * all_fm[self].shape[3])
        if not self in all_rank:
            all_rank[self] = rank
        else:
            all_rank[self] += rank

    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)

        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            handle.append(conv_tensor.register_forward_hook(forward_hook))
            handle.append(conv_tensor.register_backward_hook(backward_hook))

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer_c.zero_grad()
        data, target = data.to(cuda), target.to(cuda)
        output = model_c(data)
        loss = F.cross_entropy(output, target)
        loss.backward(retain_graph=False)


    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            values_indexes[name], indexes_dict[name] = torch.sort(torch.abs(all_rank[conv_tensor].squeeze()), descending=False)


    for h in handle:
        h.remove()
    del loss
    del output
    del all_rank
    del all_fm
    torch.cuda.empty_cache()

    return indexes_dict, values_indexes

def molchanov_improved_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    optimizer = kwargs["optimizer"]


    model_c = copy.deepcopy(model)
    optimizer_c = torch.optim.SGD(model_c.parameters(), lr=optimizer.param_groups[0]["lr"],
                                  momentum=optimizer.param_groups[0]["momentum"])


    all_fm = {}
    def forward_hook(self, input, output):
        nonlocal all_fm
        all_fm[self] = F.relu(output)

    all_rank = {}
    def backward_hook(self, grad_input, grad_output):
        nonlocal all_rank

        rank = torch.sum(all_fm[self] * grad_output[0], dim=0, keepdim=True)
        rank = rank.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        rank = rank / (all_fm[self].shape[0] * all_fm[self].shape[2] * all_fm[self].shape[3])
        if not self in all_rank:
            all_rank[self] = rank
        else:
            all_rank[self] += rank

    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            conv_tensor.register_forward_hook(forward_hook)
            conv_tensor.register_backward_hook(backward_hook)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cuda), target.to(cuda)
        output = model_c(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer_c.step()

    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(all_rank[conv_tensor].squeeze(), descending=False)

    del model_c
    del optimizer_c

    return indexes_dict, None

def molchanov_weight_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    optimizer = kwargs["optimizer"]


    model_c = copy.deepcopy(model)
    optimizer_c = torch.optim.SGD(model_c.parameters(), lr=optimizer.param_groups[0]["lr"],
                                  momentum=optimizer.param_groups[0]["momentum"])


    all_rank = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cuda), target.to(cuda)
        output = model_c(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        for name, parameters in model_c.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)

            if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
                rank = conv_tensor.weight.data * parameters.grad.data
                rank = torch.sum(rank.view(rank.shape[0], -1), dim=1)
                if not conv_tensor in all_rank:
                    all_rank[conv_tensor] = rank
                else:
                    all_rank[conv_tensor] += rank
        optimizer_c.step()


    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(torch.abs(all_rank[conv_tensor].squeeze()), descending=False)

    return indexes_dict, None

def molchanov_weight_criterion_frcnn(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    frcnn_extra = kwargs["frcnn_extra"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    optimizer = kwargs["optimizer"]

    #Hack to save previous weight before updates of molchanov_weight
    torch.save(model.state_dict(), "tmp_state_dict_for_crit.pth")

    optimizer_c = torch.optim.SGD(model.parameters(), lr=optimizer.param_groups[0]["lr"],
                                  momentum=optimizer.param_groups[0]["momentum"])




    all_rank = {}

    data_iter = iter(frcnn_extra.dataloader_train)
    for step in range(frcnn_extra.iters_per_epoch):
        data = next(data_iter)
        im_data = data[0].to(cuda)
        im_info = data[1].to(cuda)
        gt_boxes = data[2].to(cuda)
        num_boxes = data[3].to(cuda)

        #model.zero_grad()
        optimizer_c.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        loss.backward()

        for name, parameters in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)

            if parameters.requires_grad and (param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS):
                conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                rank = conv_tensor.weight.data.detach().cpu() * parameters.grad.data.detach().cpu()
                rank = torch.sum(rank.view(rank.shape[0], -1), dim=1)
                #rank = rank.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
                #rank = rank / (all_fm[conv_tensor].shape[0] * all_fm[conv_tensor].shape[2] * all_fm[conv_tensor].shape[3])
                if not conv_tensor in all_rank:
                    all_rank[conv_tensor] = rank
                else:
                    all_rank[conv_tensor] += rank

        optimizer_c.step()

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if parameters.requires_grad and (param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS):
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(torch.abs(all_rank[conv_tensor].squeeze()), descending=False)

    model.load_state_dict(torch.load("tmp_state_dict_for_crit.pth"))
    del optimizer_c

    return indexes_dict, None


def sgd_gradient_criterion(**kwargs):
    indexes_dict = {}
    model = kwargs["model"]
    train_loader = kwargs["train_loader"]
    cuda = kwargs["cuda"]
    model_adapter = kwargs["model_adapter"]
    optimizer = kwargs["optimizer"]


    all_grad_norm = {}
    model_c = copy.deepcopy(model)
    optimizer_c = torch.optim.SGD(model_c.parameters(), lr=optimizer.param_groups[0]["lr"], momentum=optimizer.param_groups[0]["momentum"])

    def backward_hook(self, grad_input, grad_output):
        nonlocal  all_grad_norm
        if hasattr(self.weight, 'grad') and self.weight.grad is not None:
            all_grad_norm[self] += self.weight.grad.data.view(self.weight.grad.data.shape[0], -1).norm(1, 1)


    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            conv_tensor.register_backward_hook(backward_hook)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cuda), target.to(cuda)
        output = model_c(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer_c.step()

    for name, parameters in model_c.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model_c, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(all_grad_norm[conv_tensor].squeeze(), descending=False)

    return indexes_dict, None

def init_from_pretrained(**kwargs):
    param_name = kwargs['param_name']
    reset_index = kwargs['reset_index']
    pretrained_model = kwargs['pretrained_model']
    in_channels_indexes = kwargs['in_channels_indexes']
    model = torch.load(pretrained_model)

    pretrained_layer = model.state_dict()[param_name]
    pretrained_index = torch.randint(high=pretrained_layer.shape[0], size=reset_index.shape, dtype=torch.long)
    if len(in_channels_indexes) == 0:
        return pretrained_layer[pretrained_index]
    return pretrained_layer[pretrained_index, in_channels_indexes[-1], :, :]

def prune_strategy(cuda, decay_rates_c, epoch, finished_list, forced_remove, get_weak_fn, initializer_fn, model,
                   model_adapter, model_architecture, optimizer, original_c, parameters_hard_removed_total,
                   prune_index_dict, remove_ratio, removed_filters_total, type_list):
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

                keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                      sorted_filters_index, forced_remove,
                                                      original_c=original_c[name], decay_rates_c=decay_rates_c[name],
                                                      epoch=epoch)
                if reset_index is not None:
                    keep_index = torch.cat((keep_index, reset_index))
                new_conv_tensor = create_conv_tensor(conv_tensor, out_channels_keep_indexes, initializer_fn,
                                                     keep_index, reset_index).to(cuda)
                model_adapter.set_layer(model, param_type, new_conv_tensor, tensor_index, layer_index, block_index)

                if name not in model_architecture:
                    model_architecture[name] = []
                model_architecture[name].append(keep_index.shape[0])

                removed_filters_total_epoch += original_out_channels - keep_index.shape[0]
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

                    keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                          sorted_filters_index, forced_remove,
                                                          original_c=original_c[name],
                                                          decay_rates_c=decay_rates_c[name], epoch=epoch)

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

                        keep_index, reset_index = get_weak_fn(original_out_channels, 0, remove_ratio,
                                                              sorted_filters_index, forced_remove,
                                                              original_c=original_c[d_name],
                                                              decay_rates_c=decay_rates_c[d_name], epoch=epoch)

                        if reset_index is not None:
                            keep_index = torch.cat((keep_index, reset_index))
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
                type_list[i], out_channels_keep_indexes[i], reset_indexes[i], None)
        else:
            index_op_dict[optimizer.param_groups[0]['params'][i]] = (
            type_list[i], out_channels_keep_indexes[i], reset_indexes[i], in_channels_keep_indexes[i])
    for k, v in index_op_dict.items():
        if v[0] == ParameterType.CNN_WEIGHTS or v[0] == ParameterType.DOWNSAMPLE_WEIGHTS:
            if v[3] is not None:
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
    removed_filters_total += removed_filters_total_epoch
    parameters_hard_removed_total += parameters_hard_removed_per_epoch
    return optimizer, parameters_hard_removed_total, parameters_reset_removed, removed_filters_total, removed_filters_total_epoch, reset_filters_total_epoch
