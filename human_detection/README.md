# 基于YOLOv5的人类检测

本仓库包含一个使用YOLOv5s.pt（PyTorch）模型实现的人类检测项目。该模型基于COCO数据集进行训练，并在一个专注于人物检测的子集数据集COCO_person上进行了进一步微调。

## 数据集

训练和评估所使用的数据集组织结构如下：

- **coco** - COCO数据集，包含注释、图像和标签。
    - **annotations** - COCO数据集的注释文件。
    - **images** - COCO数据集的图像文件。
    - **labels** - COCO数据集的标签文件。

- **coco_person** - 专注于人物检测的子集数据集。
    - **images** - 人物检测子集数据集的图像文件。
    - **labels** - 人物检测子集数据集的标签文件。
    - **extract.py** - 从COCO数据集中提取人物检测子集数据集。
    - **train2017_person.txt** - 人物检测子集的训练集文件列表。
    - **val2017_person.txt** - 人物检测子集的验证集文件列表。
    - **train2017_person_5000.txt** - 包含5000个样本的人物检测子集训练集文件列表。
    - **val2017_person_1000.txt** - 包含1000个样本的人物检测子集验证集文件列表。

## YOLOv5

- **yolov5** - 包含YOLOv5s.pt模型及其相关文件。

注意：yolov5直接从github上clone而来，但是在此基础上进行了一些修改，以适应本项目的需要。

修改1：在`yolov5/data/`中添加`person.yaml`文件，用于指定数据集的相关信息。内容如下

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: E:/wpt/Documents/CS/Yolo_coco/datasets/coco_person  # dataset root dir
train: train2017_person_5000.txt  # train images (relative to 'path')
val: val2017_person_1000.txt  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: [ 'person' ]  # class names
```

修改2：在`yolov5/data/`中添加`yolov5s_person.yaml`文件，用于指定模型的相关信息。内容如下：

```yaml
# Parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [ 10,13, 16,30, 33,23 ]  # P3/8
  - [ 30,61, 62,45, 59,119 ]  # P4/16
  - [ 116,90, 156,198, 373,326 ]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [ [ -1, 1, Conv, [ 64, 6, 2, 2 ] ],  # 0-P1/2
    [ -1, 1, Conv, [ 128, 3, 2 ] ],  # 1-P2/4
    [ -1, 3, C3, [ 128 ] ],
    [ -1, 1, Conv, [ 256, 3, 2 ] ],  # 3-P3/8
    [ -1, 6, C3, [ 256 ] ],
    [ -1, 1, Conv, [ 512, 3, 2 ] ],  # 5-P4/16
    [ -1, 9, C3, [ 512 ] ],
    [ -1, 1, Conv, [ 1024, 3, 2 ] ],  # 7-P5/32
    [ -1, 3, C3, [ 1024 ] ],
    [ -1, 1, SPPF, [ 1024, 5 ] ],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [ [ -1, 1, Conv, [ 512, 1, 1 ] ],
    [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
    [ [ -1, 6 ], 1, Concat, [ 1 ] ],  # cat backbone P4
    [ -1, 3, C3, [ 512, False ] ],  # 13

    [ -1, 1, Conv, [ 256, 1, 1 ] ],
    [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
    [ [ -1, 4 ], 1, Concat, [ 1 ] ],  # cat backbone P3
    [ -1, 3, C3, [ 256, False ] ],  # 17 (P3/8-small)

    [ -1, 1, Conv, [ 256, 3, 2 ] ],
    [ [ -1, 14 ], 1, Concat, [ 1 ] ],  # cat head P4
    [ -1, 3, C3, [ 512, False ] ],  # 20 (P4/16-medium)

    [ -1, 1, Conv, [ 512, 3, 2 ] ],
    [ [ -1, 10 ], 1, Concat, [ 1 ] ],  # cat head P5
    [ -1, 3, C3, [ 1024, False ] ],  # 23 (P5/32-large)

    [ [ 17, 20, 23 ], 1, Detect, [ nc, anchors ] ],  # Detect(P3, P4, P5)
  ]
```

## 使用方法

1. 进行数据集抽取和划分，生成文件列表。`python extract.py`。
2. 进入`yolov5`目录
3. 安装依赖项：`pip install -r requirements.txt`。
4. 训练模型：`python train.py --img 640 --batch 4 --epochs 8 --data ./data/person.yaml --cfg ./models/yolov5s_person.yaml --weights ./yolov5s.pt
`。
5. 运行检测程序：`python detect.py --source inference/images/ --weights ./runs/train/exp7/weights/best.pt
   `。
6. 根据需要，可以调整程序中的参数和路径。
