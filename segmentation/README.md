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
We train and test our structures on [CUB200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), where it gives 11788 bird pictures of 200 kinds. According to the provided train_test_split annotations, we split the pictures into training set(5994 pictures) and validation set(5794 pictures). It provide the annotation for segmentation in which it only annotate the bird zone and non-bird zone, like this 


So there are only 2 classes of zones for segmentation. We test the "classwise IoU" and "total mean IoU" for all structures base on the whole validation set. The result is as follows

| Encoder | decoder | Mean Acc | total IoU |
| ---- | ---- | ---- | ---- |
| alexnet | fcn_8 | | | 
| alexnet | fcn_32 | | |
| vgg16 | fcn_8 | | |
| vgg16 | fcn_32 | | |


for resnet, we didn't manage to finish the training because it requires way too much FLOPs and we don't have better GPU. So we quote the test result on [other people's work](https://arxiv.org/pdf/1611.08986.pdf)

| Encoder | decoder | Pixel Acc | Mean Acc | Mean IoU |
| ---- | ---- | ---- | ---- | ---- |
| resnet50 | FCN_8 | 74.42 | 47.42 | 33.89 |
| resnet101 | FCN_8 | 75.56 | 50.11 | 35.76 |
## Visualization

## How to use
we stores all the structures in [models](https://github.com/yangcyself/CS385ML/tree/master/segmentation/models), where all the FCN decoders is representated in [fcn.py](https://github.com/yangcyself/CS385ML/blob/master/segmentation/models/fcn.py).

- Download the [CUB200 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

- Unzip the dataset, place it in a specific place in your computer.

- Configure the `DATA_PATH` in the [config.py](https://github.com/yangcyself/CS385ML/blob/master/segmentation/config.py), where it should be in the format "/.../CUB_200_2011"

- Configure The `MODEL_NAME` in the [config.py](https://github.com/yangcyself/CS385ML/blob/master/segmentation/config.py), which is the model name you want to train, test or predict.

- Configure The `BATCH_SIZE`, `EPOCH` and other hyper parameters

- (optional) Download the [pretrained model]() and place it under the "checkpoints/" directory. Configure the `AUTO_RESUME` to `True`

- Go into this repository and start training or test on the following code
```
python train.py
python predict.py
```

## Summary
In this project we explored several famous network structures for FCN. We test its accuracy as well as IoU.
## Citation
- The sourse code of models and some utils are refered from https://github.com/divamgupta/image-segmentation-keras

- Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001. download pdf

- ] B. Shuai, T. Liu, and G. Wang, “Improving fully convolution network
for semantic segmentation,” arXiv preprint arXiv:1611.08986, 2016