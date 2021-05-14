# EE800 2021 Spring Wuwei Zhu
## 0. Mask Detection using YOLOv3-SE 
This project focuses on how to insert SE block into YOLOv3 and YOLOv3-tiny to help them improve their performance.
<img src="https://github.com/piggyjehong/yolov3/blob/main/Results/test_batch0_pred.jpg">

## 1. Dataset description and data pre-processing
Our customed dataset can be downloaded from: https://pan.baidu.com/s/11arYtPO3l9b4UhxYO5yI9Q 
password：y7uq 

The original dataset link: <https://www.kaggle.com/andrewmvd/face-mask-detection>.
This dataset contains 853 images belonging to the 3 classes, as well as their bounding boxes in the PASCAL VOC format.
The classes are:

- With mask
- Without mask
- Mask worn incorrectly

This dataset has the following structure:
```
  - archive
       - annotations (bouding box information in xml format)
       - images (pictures in png format)
```
A traditional PASCAL VOC structure should have the following structure:
```
  - VOCdevkit
       - VOC2007
            - Annotations (same as annotations in original dataset)
            - ImageSets (generate this folder by ourselves)
                - Main
                    - test.txt
                    - train.txt
                    - trainval.txt
                    - val.txt
            - JPEGImages(pictures in jpg format)
            - labels (generate this folder by ourselves)
```
Therefore, we have to do some data pre-processing in order to make the original dataset fit the YOLO model.
Finally, our COCO format Face Mask Detection dataset has the following structure:
```
        - voc
            - 2007_test.txt (test set path)
            - 2007_train.txt (training set path)
            - images
              - train2014 (training set pictures)
              - val2014 (test set pictures)  
            - labels
              - train2014 (training set annotations)
              - val2014 (test set annotations)  
            - template.data
            - template.names (store class names)
            - test.txt
            - train.txt
            - trainval.txt
            - val.txt
           
```
## 2. Create cfg file

YOLOv3-SE's cfg file is shown on the below.

<img src="https://github.com/piggyjehong/yolov3/blob/main/Results/Darknet53.jpg">

YOLOv3-tiny-SE cfg is shown on the below.

<img src="https://github.com/piggyjehong/yolov3/blob/main/Results/tiny.jpg">

## 3. Modify utils/parse_config.py

    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability', 'reduction', 'ratio', 'kernelsize']

## 4. Modify models.py

Add the SE class into models.py

  class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


## References
1. png2jpg.ipynb: <https://blog.csdn.net/weixin_40446557/article/details/104059660>
2. data pre-processing: <https://github.com/pprp/voc2007_for_yolo_torch>
3. YOLOv3: <https://github.com/ultralytics/yolov3>
4. 
