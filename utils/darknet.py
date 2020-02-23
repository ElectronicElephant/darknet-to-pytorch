import torch.nn as nn


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(fp):
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

        return blocks


def generate_modules(blocks):
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

        elif block['type'] == 'upsampe':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module(f'upsample_{idx}', upsample)

        elif block['type'] == 'route':
            try:
                start, end = [int(n_layer) for n_layer in block['layers'].split(',')]
            except ValueError:
                start = int(block['layers'])
                end = 0
            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx

            route = EmptyLayer()
            module.add_module(f'route_{idx}', route)

            filters = output_filters[idx + start]
            if end < 0:
                filters += output_filters[idx + end]

        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(idx), shortcut)

        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("yolo_{}".format(idx), detection)

        # End if
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


if __name__ == '__main__':
    tmp = parse_cfg('./../src/yolov3.cfg')
    blocks = generate_modules(tmp)
    print(blocks)
