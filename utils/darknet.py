from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, fp, wp=None):
        super(Darknet, self).__init__()
        self.blocks = self._parse_cfg(fp)
        self.net_info, self.module_list = self._generate_modules(self.blocks)
        assert len(self.blocks) == len(self.module_list) + 1
        assert self.net_info['height'] == self.net_info['width']

        if wp:
            self.header = None
            self.seen = None
            self.load_weights(wp)

    def load_weights(self, fp):
        # Thanks to
        # https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
        with open(fp, 'rb') as weight_file:
            # The first 5 values are header information
            # 1. Major version number
            # 2. Minor Version Number
            # 3. Subversion number
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(weight_file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(weight_file, dtype=np.float32)
            weights_length = len(weights)
            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]["type"]

                # If module_type is convolutional load weights, Otherwise ignore.
                if module_type == "convolutional":
                    model = self.module_list[i]
                    conv = model[0]
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"]) \
                        if "batch_normalize" in self.blocks[i + 1] else 0

                    if batch_normalize:
                        bn = model[1]

                        # Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()

                        # Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases
                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases
                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases
                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        # Cast the loaded weights into dims of model weights.
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        # Copy the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)

                    else:
                        # Number of biases
                        num_biases = conv.bias.numel()

                        # Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases

                        # reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Finally copy the data
                        conv.bias.data.copy_(conv_biases)

                    # Let us load the weights for the Convolutional layers
                    num_weights = conv.weight.numel()

                    # Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                    ptr = ptr + num_weights

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

            assert ptr == weights_length
            print(f'{ptr} of {weights_length} read')

    def forward(self, x):
        outputs = []  # cache for cat
        detections = []  # store the output layer
        for idx, module in enumerate(self.blocks[1:]):  # Filter out the first [net] layer
            if module['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = self.module_list[idx](x)
            elif module['type'] == 'route':
                layers = module['layers']
                if len(layers) == 0:
                    x = outputs[idx + layers[0]]
                else:
                    x = torch.cat([outputs[idx + offset] for offset in layers], 1)
            elif module['type'] == 'shortcut':
                x = outputs[idx - 1] + outputs[idx + int(module['from'])]
            elif module['type'] == 'yolo':
                anchors = self.module_list[idx][0].anchors
                input_size = int(self.net_info['height'])
                num_classes = int(module['classes'])
                detections.append(self.yolo_output_transform(x, input_size, anchors, num_classes))
            else:
                raise RuntimeError('Please report the bug.')
            # End if
            outputs.append(x)  # Cache the output

        return torch.cat(detections, 1)

    def get_results(self, x, num_classes=80, conf_thres=0.5, nms_thres=0.4):
        # TorchVision only provides a basic nms function, so I decided to write my own.
        raw_preds = x.detach()

        # Transform to (x, y, x, y) format
        box_corner = raw_preds.new(raw_preds.shape)
        box_corner[:, :, 0] = (raw_preds[:, :, 0] - raw_preds[:, :, 2] / 2)
        box_corner[:, :, 1] = (raw_preds[:, :, 1] - raw_preds[:, :, 3] / 2)
        box_corner[:, :, 2] = (raw_preds[:, :, 0] + raw_preds[:, :, 2] / 2)
        box_corner[:, :, 3] = (raw_preds[:, :, 1] + raw_preds[:, :, 3] / 2)
        box_corner = box_corner[:, :, 0:4]

        raw_classes = raw_preds[:, :, 5:]
        assert raw_classes.size(2) == num_classes
        classes = torch.argmax(raw_classes, 2)
        raw_preds = raw_preds[:, :, 0:6]
        raw_preds[:, :, 0:4] = box_corner
        raw_preds[:, :, 5] = classes

        preds = []
        for raw_preds_batch in raw_preds:
            # conf_thresold
            batch_preds = raw_preds_batch[raw_preds_batch[:, 4] > conf_thres]

            # nms
            obj_idx = nms(batch_preds[:, 0:4], batch_preds[:, 4], iou_threshold=nms_thres)
            preds_after_nms = batch_preds[obj_idx]

            preds.append(preds_after_nms)

        return preds

    def summary(self):
        print(self.net_info)
        print(self.module_list)

    @staticmethod
    def _parse_cfg(fp):
        """
        Parses the darknet config file

        :return a list of dicts, each of which represents a block
        """

        with open(fp) as cfg:
            lines = [x.strip() for x in cfg.readlines()]  # Filter out all '\n's
            lines = [x for x in lines if len(x) > 0 and x[0] != '#']  # Filter out the comments and blank lines
            # print(lines)
            assert lines[0][0] == '['

            blocks = []
            block = {}
            for line in lines:
                if line[0] == '[':
                    if len(block) != 0:
                        blocks.append(block)
                    block = {'type': line[1:-1]}
                else:
                    key_, value_ = line.split('=')
                    block[key_.strip()] = value_.strip()

            blocks.append(block)
            return blocks

    @staticmethod
    def _generate_modules(blocks):
        assert blocks[0]['type'] == 'net'
        net_info = blocks[0]
        module_list = nn.ModuleList()

        prev_filters = 3  # start with RGB
        output_filters = []  # track the filters

        for idx, block in enumerate(blocks[1:]):
            module = nn.Sequential()

            if block['type'] == 'convolutional':
                # Interpreting the block info
                activation = block['activation']
                filters = int(block['filters'])
                padding = int(block['pad'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                pad = (kernel_size - 1) // 2 if padding else 0
                if 'batch_normalize' in block:
                    batch_normalize = int(block['batch_normalize'])
                    bias = False
                else:
                    batch_normalize = 0
                    bias = True

                # Add layers
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module(f'conv_{idx}', conv)

                if batch_normalize:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f'bn_{idx}', bn)

                if activation == 'leaky':
                    lrelu = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f'leaky_{idx}', lrelu)
                else:
                    assert activation == 'linear'

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                upsample = nn.Upsample(scale_factor=2, mode='bilinear')
                module.add_module(f'upsample_{idx}', upsample)

            elif block['type'] == 'route':
                layers = [int(n_layer) for n_layer in block['layers'].split(',')]
                layers = list(map(lambda x: x - idx if x > 0 else x, layers))
                block['layers'] = layers  # Warning: Decorator

                route = EmptyLayer()
                module.add_module(f'route_{idx}', route)

                filters = 0
                for n_layer in layers:
                    assert n_layer < 0
                    filters += output_filters[idx + n_layer]

            elif block['type'] == 'shortcut':
                shortcut = EmptyLayer()
                module.add_module('shortcut_{}'.format(idx), shortcut)

            elif block['type'] == 'yolo':
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]

                anchors = block['anchors'].split(',')
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module('yolo_{}'.format(idx), detection)

            elif block['type'] == 'maxpool':
                stride = int(block['stride'])
                size = int(block['size'])
                # Calculate the padding to keep the size
                if stride == 1:
                    # Should Keep Size
                    # assert size % 2 == 1  # Otherwise, the padding can not be inferred
                    padding = ceil((size - 2) / 2)
                    if padding == 0:  # Dirty patch for yolov3-tiny
                        padding = 1
                        assert size == 2
                        size += 1
                else:
                    padding = 0
                maxpool = nn.MaxPool2d(size, stride=stride, padding=padding)
                module.add_module(f'maxpool_{idx}', maxpool)

            else:
                raise NotImplementedError(f"{block['type']} Not Supported!")

            # End if
            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

        return net_info, module_list

    @staticmethod
    def yolo_output_transform(pred, input_size, anchors, num_classes):
        batch_size = pred.size(0)
        if input_size % pred.size(2) != 0:
            raise Warning(f'{input_size} // {pred.size(2)}')
        stride = input_size // pred.size(2)
        if input_size % stride != 0:
            raise Warning(f'{input_size} // {stride}')
        grid_size = input_size // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        # (B, anchor_x, anchor_y, 3 * (5 + C)) -> TODO
        pred = pred.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        pred = pred.transpose(1, 2).contiguous()
        pred = pred.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

        # Change anchors from original size to feature map size
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

        # Sigmoid the  centre_X, centre_Y. and object confidence
        pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])  # x
        pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])  # y
        pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])  # confidence

        # Add the center offsets
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.tensor(a, dtype=torch.float).view(-1, 1)
        y_offset = torch.tensor(b, dtype=torch.float).view(-1, 1)
        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        pred[:, :, :2] += x_y_offset

        # log space transform height and the width
        anchors = torch.tensor(anchors, dtype=torch.float)
        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        pred[:, :, 2:4] = torch.exp(pred[:, :, 2:4]) * anchors

        # Class scores
        pred[:, :, 5: 5 + num_classes] = torch.sigmoid((pred[:, :, 5: 5 + num_classes]))

        # Resize back to the original shape
        pred[:, :, :4] *= stride

        return pred
