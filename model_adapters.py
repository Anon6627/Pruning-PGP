from utils import ParameterType, int_from_str

LAYER_STR = "layer"
WEIGHT_STR = "weight"
FEATURES_STR = "features"
CONV_STR = "conv"
BN_STR = "bn"
DOWNSAMPLE_STR = "downsample"
FC_STR = "fc"
SHORTCUT_STR = "shortcut"
RCNN_BASE_STR = 'RCNN_base'
RCNN_RPN_STR = 'RCNN_rpn'
RCNN_TOP_STR = 'RCNN_top'
BIAS_STR = "bias"

class AlexNetAdapter(object):
    def __init__(self):
        pass

    def get_param_type_and_layer_index(self, param_name):
        type = None
        if WEIGHT_STR in param_name:
            if FEATURES_STR in param_name:
                type = ParameterType.CNN_WEIGHTS
            else:
                type = ParameterType.FC_WEIGHTS
        elif BIAS_STR in param_name:
            if FEATURES_STR in param_name:
                type = ParameterType.CNN_BIAS
            else:
                type = ParameterType.FC_BIAS
        return type, int(param_name.split(".")[1]), -1, -1

    def get_layer(self, model, param_type, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS:
            return model.features[conv_index]
        elif param_type == ParameterType.CNN_BIAS:
            return model.features[conv_index].bias
        elif param_type == ParameterType.FC_WEIGHTS:
            return model.classifier[conv_index]
        elif param_type == ParameterType.FC_BIAS:
            return model.classifier[conv_index].bias
        return None

    def set_layer(self, model, param_type, new_layer, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.CNN_BIAS:
            model.features[conv_index] = new_layer
        elif param_type == ParameterType.FC_WEIGHTS or param_type == ParameterType.FC_BIAS:
            model.classifier[conv_index] = new_layer

    def num_tensor_parameters(self, model):
        return len(model.features) + len(model.classifier)

class VGGRCNNAdapter(object):
    def __init__(self):
        #self.batch_norm_index = [1, 4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41, 44, 47, 50]
        self.last_layer_index = 28
        pass

    def get_param_type_and_layer_index(self, param_name):
        type = None

        if RCNN_BASE_STR in param_name:
            if WEIGHT_STR in param_name:
                type = ParameterType.CNN_WEIGHTS
            else:
                type = ParameterType.CNN_BIAS

        tensor_index = -1
        if (type == ParameterType.CNN_WEIGHTS or type == ParameterType.CNN_BIAS) and len(int_from_str(param_name)) != 0:
            tensor_index = int_from_str(param_name)[0]

        return type, tensor_index, -1, -1

    def get_layer(self, model, param_type, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS:
            return model.RCNN_base[conv_index]
        elif param_type == ParameterType.CNN_BIAS:
            return model.RCNN_base[conv_index].bias

        return None

    def set_layer(self, model, param_type, new_layer, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.CNN_BIAS:
            model.RCNN_base[conv_index] = new_layer

    def num_tensor_parameters(self, model):
        return len(model.features) + len(model.classifier)

class VGG16Adapter(object):
    def __init__(self):
        self.batch_norm_index = [1, 4, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 41, 44, 47, 50]
        pass

    def get_param_type_and_layer_index(self, param_name):
        type = None
        if WEIGHT_STR in param_name:
            if FEATURES_STR in param_name:
                type = ParameterType.CNN_WEIGHTS
            else:
                type = ParameterType.FC_WEIGHTS
        elif BIAS_STR in param_name:
            if FEATURES_STR in param_name:
                type = ParameterType.CNN_BIAS
            else:
                type = ParameterType.FC_BIAS
        tensor_index = -1
        if len(int_from_str(param_name)) != 0:
            tensor_index = int_from_str(param_name)[0]
            if tensor_index in self.batch_norm_index:
                if WEIGHT_STR in param_name:
                    type = ParameterType.BN_WEIGHT
                else:
                    type = ParameterType.BN_BIAS
        return type, tensor_index, -1, -1

    def get_layer(self, model, param_type, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS:
            return model.features[conv_index]
        elif param_type == ParameterType.CNN_BIAS:
            return model.features[conv_index].bias
        elif param_type == ParameterType.FC_WEIGHTS:
            return model.classifier
        elif param_type == ParameterType.FC_BIAS:
            return model.classifier.bias
        elif param_type == ParameterType.BN_WEIGHT:
            return model.features[conv_index]
        elif param_type == ParameterType.BN_BIAS:
            return model.features[conv_index].bias
        return None

    def set_layer(self, model, param_type, new_layer, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS or param_type == ParameterType.CNN_BIAS:
            model.features[conv_index] = new_layer
        elif param_type == ParameterType.FC_WEIGHTS or param_type == ParameterType.FC_BIAS:
            model.classifier = new_layer
        elif param_type == ParameterType.BN_WEIGHT or param_type == ParameterType.BN_BIAS:
            model.features[conv_index] = new_layer

    def num_tensor_parameters(self, model):
        return len(model.features) + len(model.classifier)

class ResNetAdapter(object):
    def __init__(self):
        pass

    def conv_or_bn_type(self, param_name):
        if CONV_STR in param_name:
            return ParameterType.CNN_WEIGHTS
        elif BN_STR in param_name:
            if WEIGHT_STR in param_name:
                return ParameterType.BN_WEIGHT
            else:
                return ParameterType.BN_BIAS
        elif DOWNSAMPLE_STR in param_name:
            downsample_index = int_from_str(param_name)[2]

            if downsample_index == 0:
                if WEIGHT_STR in param_name:
                    return ParameterType.DOWNSAMPLE_WEIGHTS
                else:
                    return ParameterType.DOWNSAMPLE_BIAS
            else:
                if WEIGHT_STR in param_name:
                    return ParameterType.DOWNSAMPLE_BN_W
                else:
                    return ParameterType.DOWNSAMPLE_BN_B
        elif FC_STR in param_name:
            return ParameterType.FC_WEIGHTS

    def get_downsample(self, model, layer_index, block_index):
        layer_key = LAYER_STR + str(layer_index)
        block_list = model._modules[layer_key]
        block = block_list[block_index]._modules
        tkey = self.type_2_str(ParameterType.DOWNSAMPLE_WEIGHTS)
        if DOWNSAMPLE_STR in block:
            name = LAYER_STR + str(layer_index) + "." + str(block_index) + "." + DOWNSAMPLE_STR + ".0.weight"
            return block[tkey][0], name
        else:
            return None, ""

    def get_conv2_from_downsample(self, model, layer_index, block_index):
        layer_key = LAYER_STR + str(layer_index)
        block_list = model._modules[layer_key]
        block = block_list[block_index]._modules
        tkey = self.type_2_str(ParameterType.CNN_WEIGHTS)
        tensor_key = tkey + str(2)
        tensor = block[tensor_key]

        name = "{}.{}.{}.weight".format(layer_key, block_index, tensor_key)

        return tensor, name

    def get_param_type_and_layer_index(self, param_name):
        type = None
        layer_index = -1
        block_index = -1
        tensor_index = -1
        if not LAYER_STR in param_name:
            type = self.conv_or_bn_type(param_name)
            if type != ParameterType.FC_WEIGHTS:
                tensor_index = int_from_str(param_name)[0]
        else:
            layer_index, block_index, tensor_index = int_from_str(param_name)
            type = self.conv_or_bn_type(param_name)
        return type, tensor_index, layer_index, block_index

    def type_2_str(self, param_type):
        if param_type == ParameterType.CNN_WEIGHTS:
            return CONV_STR
        elif param_type == ParameterType.BN_WEIGHT:
            return BN_STR
        elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_BN_W:
            return DOWNSAMPLE_STR
        return None

    def get_layer(self, model, param_type, tensor_index, layer_index, block_index):
        if layer_index == -1:
            if param_type == ParameterType.FC_WEIGHTS:
                return model._modules[FC_STR]
            tensor_key = self.type_2_str(param_type) + str(tensor_index)
            tensor = model._modules[tensor_key]
            return tensor

        layer_key = LAYER_STR + str(layer_index)
        block_list = model._modules[layer_key]
        block = block_list[block_index]._modules
        tkey = self.type_2_str(param_type)
        tensor_key = tkey + str(tensor_index)
        if tkey == DOWNSAMPLE_STR:
            tensor_key = tensor_key[:-1]

        tensor = block[tensor_key]

        if param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            tensor = tensor[0]
        if param_type == ParameterType.DOWNSAMPLE_BN_W:
            tensor = tensor[1]
        elif param_type == ParameterType.DOWNSAMPLE_BN_B:
            tensor = tensor[1]
        return tensor

    def set_layer(self, model, param_type, new_layer, tensor_index, layer_index, block_index):
        if layer_index == -1:
            if param_type == ParameterType.FC_WEIGHTS:
                model._modules[FC_STR] = new_layer
            else:
                tensor_key = self.type_2_str(param_type) + str(tensor_index)
                model._modules[tensor_key] = new_layer
        else:
            layer_key = LAYER_STR + str(layer_index)
            tensor_key = self.type_2_str(param_type) + str(tensor_index)
            if param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                model._modules[layer_key][block_index]._modules[tensor_key[:-1]][0] = new_layer
            elif param_type == ParameterType.DOWNSAMPLE_BN_W:
                model._modules[layer_key][block_index]._modules[tensor_key[:-1]][1] = new_layer
            else:
                model._modules[layer_key][block_index]._modules[tensor_key] = new_layer

class Valid_ResNetAdapter(object):
    def __init__(self):
        pass

    def conv_or_bn_type(self, param_name):
        if CONV_STR in param_name:
            return ParameterType.CNN_WEIGHTS
        elif BN_STR in param_name:
            if WEIGHT_STR in param_name:
                return ParameterType.BN_WEIGHT
            else:
                return ParameterType.BN_BIAS
        elif DOWNSAMPLE_STR in param_name:
            downsample_index = int_from_str(param_name)[2]

            if downsample_index == 0:
                if WEIGHT_STR in param_name:
                    return ParameterType.DOWNSAMPLE_WEIGHTS
                else:
                    return ParameterType.DOWNSAMPLE_BIAS
            else:
                if WEIGHT_STR in param_name:
                    return ParameterType.DOWNSAMPLE_BN_W
                else:
                    return ParameterType.DOWNSAMPLE_BN_B
        elif FC_STR in param_name:
            return ParameterType.FC_WEIGHTS

    def get_downsample(self, model, layer_index, block_index):
        layer_key = LAYER_STR + str(layer_index)
        block_list = model._modules[layer_key]
        block = block_list[block_index]._modules
        tkey = self.type_2_str(ParameterType.DOWNSAMPLE_WEIGHTS)
        if DOWNSAMPLE_STR in block:
            name = ""
            if 'Lambda' in str(block[DOWNSAMPLE_STR]) or len(block[DOWNSAMPLE_STR]) == 0:
                return None, name
            return block[DOWNSAMPLE_STR], name
        else:
            return None, ""

    def get_param_type_and_layer_index(self, param_name):
        type = None
        layer_index = -1
        block_index = -1
        tensor_index = -1
        if not LAYER_STR in param_name:
            type = self.conv_or_bn_type(param_name)
            if type != ParameterType.FC_WEIGHTS:
                tensor_index = int_from_str(param_name)[0]


        else:
            layer_index, block_index, tensor_index = int_from_str(param_name)
            type = self.conv_or_bn_type(param_name)
        return type, tensor_index, layer_index, block_index

    def type_2_str(self, param_type):
        if param_type == ParameterType.CNN_WEIGHTS:
            return CONV_STR
        elif param_type == ParameterType.BN_WEIGHT:
            return BN_STR
        elif param_type == ParameterType.DOWNSAMPLE_WEIGHTS or param_type == ParameterType.DOWNSAMPLE_BN_W:
            return DOWNSAMPLE_STR
        return None

    def get_layer(self, model, param_type, tensor_index, layer_index, block_index):
        if layer_index == -1:
            if param_type == ParameterType.FC_WEIGHTS:
                return model._modules[FC_STR]
            tensor_key = self.type_2_str(param_type) + str(tensor_index)
            tensor = model._modules[tensor_key]
            return tensor

        layer_key = LAYER_STR + str(layer_index)
        block_list = model._modules[layer_key]
        block = block_list[block_index]._modules
        tkey = self.type_2_str(param_type)
        tensor_key = tkey + str(tensor_index)
        if tkey == DOWNSAMPLE_STR:
            tensor_key = tensor_key[:-1]

        tensor = block[tensor_key]

        if param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
            tensor = tensor[0]
        if param_type == ParameterType.DOWNSAMPLE_BN_W:
            tensor = tensor[1]
        elif param_type == ParameterType.DOWNSAMPLE_BN_B:
            tensor = tensor[1]
        return tensor

    def set_layer(self, model, param_type, new_layer, tensor_index, layer_index, block_index):
        if layer_index == -1:
            if param_type == ParameterType.FC_WEIGHTS:
                model._modules[FC_STR] = new_layer
            else:
                tensor_key = self.type_2_str(param_type) + str(tensor_index)
                model._modules[tensor_key] = new_layer
        else:
            layer_key = LAYER_STR + str(layer_index)
            tensor_key = self.type_2_str(param_type) + str(tensor_index)
            if param_type == ParameterType.DOWNSAMPLE_WEIGHTS:
                model._modules[layer_key][block_index]._modules[tensor_key[:-1]][0] = new_layer
            elif param_type == ParameterType.DOWNSAMPLE_BN_W:
                model._modules[layer_key][block_index]._modules[tensor_key[:-1]][1] = new_layer
            else:
                model._modules[layer_key][block_index]._modules[tensor_key] = new_layer

class EasyNetAdapter(object):
    def __init__(self):
        pass

    def get_param_type_and_layer_index(self, param_name):
        type = None
        conv_index = 0
        if "conv1" in param_name:
            type = ParameterType.CNN_WEIGHTS
            conv_index = 1
        elif "conv2" in param_name:
            type = ParameterType.CNN_WEIGHTS
            conv_index = 2
        elif "fc1" in param_name:
            type = ParameterType.FC_WEIGHTS
            conv_index = 1
        elif "fc2" in param_name:
            type = ParameterType.FC_WEIGHTS
            conv_index = 2

        return type, conv_index, -1, -1

    def get_layer(self, model, param_type, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS and conv_index == 1:
            return model.conv1
        elif param_type == ParameterType.CNN_WEIGHTS and conv_index == 2:
            return model.conv2
        elif param_type == ParameterType.FC_WEIGHTS:
            return model.fc1
        return None

    def set_layer(self, model, param_type, new_layer, conv_index, layer_index=-1, block_index=-1):
        if param_type == ParameterType.CNN_WEIGHTS and conv_index == 1:
            model.conv1 = new_layer
        elif param_type == ParameterType.CNN_WEIGHTS and conv_index == 2:
            model.conv2 = new_layer
        elif param_type == ParameterType.FC_WEIGHTS:
            model.fc1 = new_layer

    def num_tensor_parameters(self, model):
        return 4