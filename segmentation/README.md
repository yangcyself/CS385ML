# Image Segmentation Report

This reporsitory explore **Fully Convolutional Neural Networks(FCN)** for sementatic segmentaion. We adjust several famous network structures for FCN encoder, they are mainly [alexnet](), [vgg16](), [resnet50](). For each network structure, we arrange 2 kind of FCN decoders for it. So There is totally 6 network structures for FCN.

## Structures
Generally speaking, FCN transforms the fully connected layers in traditional CNN into convolution layer. Then it add the Deconvlutional layers in the 

The detailed structures is in the following list
| Encoder | decoder | structure |
| ---- | ---- | ---- |
| alexnet | fcn_8 | [fcn_8_alexnet]()|
| alexnet | fcn_32 | [fcn_32_alexnet]()|
| vgg16 | fcn_8 | [fcn_8_vgg]()|
| vgg16 | fcn_32 | [fcn_32_vgg]()|
| resnet50 | fcn_8 | [resnet50_8_vgg]()|
| resnet50 | fcn_32 | [resnet50_32_vgg]()|

## Performance

## Visualization

## How to use

## Summary


## Citation