# 项目名称

基于词袋模型的图像检索项目

## 项目简介

该项目是一个基于词袋模型的图像检索系统。通过提取图像的特征向量并使用词袋模型对图像进行表示，可以对给定的查询图像进行相似图像的检索。

## 项目结构

```commandline
├── dataset
│   └── oxbuild_images
├── src
│   ├── train.py
│   ├── query.py
│   └── test.ipynb # 包含了解释算法原理的示例代码
├── model
│   └── model.pkl
├── result
├── README.md
└── requirements.txt
```


## 使用方法

1. 准备数据集：将要查询的图像和训练集图像放置在 `dataset/oxbuild_images` 目录中。
2. 训练模型：运行 `python train.py -dataset 数据集路径 -save 保存模型的路径` 脚本来训练模型并生成模型文件 `model/model.pkl`。
3. 执行查询：运行 `python query.py -model 模型路径 -query 查询图像路径` 脚本，选择一张查询图像进行相似图像检索，并得到结果。
4. 结果展示：将展示与查询图像最相似的20张图片，如`result/result.png`所示。

## 依赖环境

- Python 3.11
- 依赖库和版本号请参考 `requirements.txt` 文件

## 安装依赖

1. 确保已安装 Python 3.11
2. 运行以下命令安装项目依赖：

    ```pip install -r requirements.txt```
