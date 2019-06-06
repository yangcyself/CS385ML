# Image Segmentation Report

This reporsitory explore **Fully Convolutional Neural Networks(FCN)** for sementatic segmentaion. We adjust several famous network structures for FCN.

## Structures
Generally, FCN is considered as a combination of encoders and decoders. The encoders is adjusted by traditional CNNs, where FCN transforms the fully connected layers into convolutional layers. For The decoder, it combines the feature maps of the last layer and some middle layers of the encoder and restore them to the same size of the input image, so as to generate a prediction for each pixel and retain the spatial information in the original input image.
![from https://blog.csdn.net/qq_36269513/article/details/80420363](https://img-blog.csdn.net/20160508234037674)
Based on some previous works of other people, we explored several structures of FCN. For the encoder, we explored **alexnet, vgg16,** and **resnet50**
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