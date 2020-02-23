# darknet-to-pytorch
Easily convert Darknet weights with cfgs to PyTorch!

Currently only support `yolov3`, `yolov3-spp`, and `yolov3-tiny`, since I don't think anyone would use `Darknet` to implement `RNNs`.

# Getting the weights and cfgs
```
cd src
bash get_weights.sh
```

# Credits
- https://blog.paperspace.com/tag/series-yolo/
- https://pjreddie.com/darknet/yolo/
