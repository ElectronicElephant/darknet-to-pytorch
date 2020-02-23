# darknet-to-pytorch
Easily convert Darknet weights with cfgs to PyTorch!

Currently only support `yolov3`, `yolov3-spp`, and `yolov3-tiny`,
since I don't think anyone would use `Darknet` to implement `RNNs`.

# Contribute
Currently, I don't know how `Darknet C++ version` pads the feature map to
keep the size after max-pooling when `stride = 1`.

Thus, loading official weights of `yolov3-spp` and `yolov3-tiny` may cause problem.

If you do know a bit, feel free to open an issue and tell me. Thank you.

# Getting the weights and cfgs
```
cd src
bash get_weights.sh
```

# Credits
- https://blog.paperspace.com/tag/series-yolo/
- https://pjreddie.com/darknet/yolo/
