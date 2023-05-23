# python query.py -model 模型路径 -query 查询图像路径

import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse


# 保存K均值聚类模型和词袋特征
class Model:
    def __init__(self, kmeans, bag_of_features, images_path):
        self.kmeans = kmeans
        self.bag_of_features = bag_of_features
        self.num_words = kmeans.n_clusters
        self.image_paths = images_path


# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image retrieval using OpenCV')
    parser.add_argument('-model', type=str, help='Path to the trained model')
    parser.add_argument('-query', type=str, help='Path to the query image')
    return parser.parse_args()


def main(model_path, query_path):
    # 加载模型
    model = joblib.load(model_path)
    kmeans = model.kmeans
    bag_of_features = model.bag_of_features
    images_path = model.image_paths

    # 初始化SIFT特征提取器
    sift = cv2.SIFT_create()

    # 读取查询图像，并提取SIFT特征
    query_img = cv2.imread(query_path)
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    query_keypoints, query_descriptor = sift.detectAndCompute(query_gray, None)
    # 使用K均值聚类将查询图像的描述符映射到词袋中心
    query_labels = kmeans.predict(query_descriptor)
    # 统计每个词袋中心的频次
    query_hist, _ = np.histogram(query_labels, bins=np.arange(101))
    # 归一化特征向量
    query_hist = query_hist.astype(float)
    query_hist /= np.sum(query_hist)

    # 计算查询图像与数据集中每张图像的相似度
    scores = np.sum(np.minimum(bag_of_features, query_hist), axis=1)

    # 获取相似度最高的前20张图像的索引
    top20 = np.argsort(scores)[::-1][:20]

    # 可视化结果
    plt.figure('Image Retrieval using OpenCV')
    plt.subplot(5, 5, 1)
    plt.title('Query Image')
    plt.imshow(query_img[:, :, ::-1])
    plt.axis('off')
    for i in range(20):
        img = cv2.imread(images_path[top20[i]])
        plt.subplot(5, 5, i + 6)
        plt.imshow(img)
        plt.title('Similar %d' % (i + 1))
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.model, args.query)
