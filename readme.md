# Orthographic Feature Transform for Monocular 3D Object Detection

![OFTNet-Architecture](https://github.com/tom-roddick/oft/raw/master/architecture.png "OFTNet-Architecture")
This is a PyTorch implementation of the OFTNet network from the paper [Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/abs/1811.08188). The code currently supports training the network from scratch on the KITTI dataset - intermediate results can be visualised using Tensorboard. The current version of the code is intended primarily as a reference, and for now does not support decoding the network outputs into bounding boxes via non-maximum suppression. This will be added in a future update. Note also that there are some slight implementation differences from the original code used in the paper. Please see `train.py` for details of training options.


## Citation
If you find this work useful please cite the paper using the citation below.
```
@article{roddick2018orthographic,  
  title={Orthographic feature transform for monocular 3d object detection},  
  author={Roddick, Thomas and Kendall, Alex and Cipolla, Roberto},  
  journal={British Machine Vision Conference},  
  year={2019}  
}
```