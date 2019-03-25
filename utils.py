import torch
from enum import Enum
import re

import torch.nn.functional as F

class LoggerForSacred():
    def __init__(self, visdom_logger, ex_logger=None):
        self.visdom_logger = visdom_logger
        self.ex_logger = ex_logger


    def log_scalar(self, metrics_name, value, step):
        self.visdom_logger.scalar(metrics_name, step, [value])
        if self.ex_logger is not None:
            self.ex_logger.log_scalar(metrics_name, value, step)

class ParameterType(Enum):
    CNN_WEIGHTS = 1
    CNN_BIAS = 2
    FC_WEIGHTS = 3
    FC_BIAS = 4
    BN_WEIGHT = 5
    BN_BIAS = 6
    DOWNSAMPLE_WEIGHTS = 7
    DOWNSAMPLE_BIAS = 8
    DOWNSAMPLE_BN_W = 9
    DOWNSAMPLE_BN_B = 10

def int_from_str(str):
    return list(map(int, re.findall(r'\d+', str)))


def eval(model, device, test_loader, is_break=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if is_break:
                break

    acc = 100. * correct / len(test_loader.dataset)
    del output
    #torch.cuda.empty_cache()

    return acc

def train(model, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += float(loss.item())
        loss.backward(retain_graph=False)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_break:
            break
        
    del loss
    del output
    #torch.cuda.empty_cache()
    return total_loss / len(train_loader)

def mtrain(model, model_adapter, optimizer, device, train_loader, is_break=False):
    indexes_dict = {}
    values_indexes = {}
    handle = []

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

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)

        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            handle.append(conv_tensor.register_forward_hook(forward_hook))
            handle.append(conv_tensor.register_backward_hook(backward_hook))

    total_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += float(loss.item())
        loss.backward(retain_graph=False)

        optimizer.step()
        if batch_idx == 2 and is_break:
            break


    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            values_indexes[name], indexes_dict[name] = torch.sort(torch.abs(all_rank[conv_tensor].squeeze()), descending=False)


    for h in handle:
        h.remove()
    del loss
    del output
    del all_rank
    del all_fm
    torch.cuda.empty_cache()

    return total_loss, indexes_dict

def gngtrain(model, model_adapter, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()

    all_grad_norm = {}

    def backward_hook(self, grad_input, grad_output):
        nonlocal  all_grad_norm

        if hasattr(self.weight, 'grad') and self.weight.grad is not None:
            if not self in all_grad_norm:
                all_grad_norm[self] = self.weight.grad.data.view(self.weight.grad.data.shape[0], -1).norm(1, 1)
            else:
                all_grad_norm[self] += self.weight.grad.data.view(self.weight.grad.data.shape[0], -1).norm(1, 1)

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            conv_tensor.register_backward_hook(backward_hook)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if batch_idx == 2 and is_break:
            break

    indexes_dict = {}
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(all_grad_norm[conv_tensor].squeeze(), descending=False)

    return total_loss / len(train_loader), indexes_dict

def gntrain_not_sure(model, model_adapter, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()

    all_grad_norm = {}

    def backward_hook(self, grad_input, grad_output):
        nonlocal  all_grad_norm

        if hasattr(self.weight, 'grad') and self.weight.grad is not None:
            if not self in all_grad_norm:
                all_grad_norm[self] = self.weight.grad.data.view(self.weight.grad.data.shape[0], -1)
            else:
                all_grad_norm[self] += self.weight.grad.data.view(self.weight.grad.data.shape[0], -1)

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            conv_tensor.register_backward_hook(backward_hook)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if batch_idx == 2 and is_break:
            break

    indexes_dict = {}
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(torch.abs(all_grad_norm[conv_tensor].sum(dim=1)).squeeze(), descending=False)

    return total_loss / len(train_loader), indexes_dict

def gnstrain(model, model_adapter, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()

    all_grad_norm = {}

    def backward_hook(self, grad_input, grad_output):
        nonlocal  all_grad_norm

        if hasattr(self.weight, 'grad') and self.weight.grad is not None:
            if not self in all_grad_norm:
                all_grad_norm[self] = self.weight.grad.data.view(self.weight.grad.data.shape[0], -1)
            else:
                all_grad_norm[self] += self.weight.grad.data.view(self.weight.grad.data.shape[0], -1)

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            conv_tensor.register_backward_hook(backward_hook)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if batch_idx == 2 and is_break:
            break

    indexes_dict = {}
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(all_grad_norm[conv_tensor].norm(1, 1).squeeze(), descending=False)

    return total_loss / len(train_loader), indexes_dict

def twtrain(model, model_adapter, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()

    all_rank = {}


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
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
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_break:
            break

    indexes_dict = {}
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(torch.abs(all_rank[conv_tensor].squeeze()), descending=False)

    return total_loss / len(train_loader), indexes_dict

def l2train(model, model_adapter, optimizer, device, train_loader, is_break=False):
    total_loss = 0.

    # One epoch step gradient for target
    optimizer.zero_grad()
    model.train()

    all_rank = {}


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()
        for name, parameters in model.named_parameters():
            param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)

            if parameters.requires_grad and (param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS):
                conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
                rank = parameters.data.view(parameters.shape[0], -1).norm(dim=1, p=2)
                #rank = rank.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
                #rank = rank / (all_fm[conv_tensor].shape[0] * all_fm[conv_tensor].shape[2] * all_fm[conv_tensor].shape[3])
                if not conv_tensor in all_rank:
                    all_rank[conv_tensor] = rank
                else:
                    all_rank[conv_tensor] += rank
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        if is_break:
            break

    indexes_dict = {}
    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            conv_tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)
            _, indexes_dict[name] = torch.sort(all_rank[conv_tensor].squeeze(), descending=False)

    return total_loss / len(train_loader), indexes_dict