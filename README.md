# EE800 2021 Spring Wuwei Zhu
## 0. Mask Detection using YOLOv3-SE 
This project focuses on how to insert SE block into YOLOv3 and YOLOv3-tiny to help them improve their performance.

## 1. Dataset description and data pre-processing
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
  - coco
       
            - 2007_test.txt
            - 2007_train.txt
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
## References
1. png2jpg.ipynb: <https://blog.csdn.net/weixin_40446557/article/details/104059660>
2. data pre-processing: <https://github.com/pprp/voc2007_for_yolo_torch>
3. YOLOv3: <https://github.com/ultralytics/yolov3>
4. 
