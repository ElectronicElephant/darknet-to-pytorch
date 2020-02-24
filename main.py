import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
from torchvision.transforms.functional import to_tensor, resize

from utils.darknet import Darknet


def get_net(net_name):
    if net_name == 'yolov3':
        return Darknet('./src/yolov3.cfg', './src/yolov3.weights')
    elif net_name == 'yolov3-spp':
        return Darknet('./src/yolov3-spp.cfg', './src/yolov3-spp.weights')
    elif net_name == 'yolov3-tiny':
        return Darknet('./src/yolov3-tiny.cfg', './src/yolov3-tiny.weights')
    else:
        raise NotImplementedError('%s is not supported.'
                                  'Currently only support yolov3, yolov3-spp, or yolov3-tiny'
                                  % net_name)


def load_test_img(fp, size):
    assert isinstance(fp, str)

    image = Image.open(fp)
    image = resize(image, (size, size))
    x = to_tensor(image)
    x.unsqueeze_(0)
    print(x.shape)

    return image, x


def print_preds(preds, image):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    COLORS = {i: plt.get_cmap('hsv')(i / len(preds[0]))
              for i in range(len(preds[0]))}

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(image)

    for idx, pred in enumerate(preds[0]):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (int(pred[0]), int(pred[1])), int(pred[2] - pred[0]), int(pred[3] - pred[1]), linewidth=1,
            edgecolor=COLORS[idx], facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        class_name = CLASSES[int(pred[5])]
        score = '{:.3f}'.format(pred[4])
        if class_name or score:
            ax.text(pred[0], pred[1] - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(alpha=0.5),
                    fontsize=12, color='white')
    plt.show()


if __name__ == '__main__':
    net = get_net('yolov3-spp')
    net.summary()

    size = int(net.net_info['height'])
    img, x = load_test_img('./src/dog-cycle-car.png', size)

    raw_preds = net(x).detach()

    preds = net.get_results(raw_preds, num_classes=80, conf_thres=0.5, nms_thres=0.4)
    print(preds)

    print_preds(preds, img)
