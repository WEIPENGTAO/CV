"""
用PCA+KNN算法实现人脸识别
（1）数据集自选；orl_faces
（2）详细说明实验参数，包括参与训练和测试的图片数量，分类个数，降维维度，knn参数等，总结各参数对准确率的影响，不断提高准确率。
"""
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# 读取数据集
def load_dataset():
    # 初始化数据集和标签数组
    data = []
    label = []

    # 读取数据集
    for i in range(40):
        for j in range(10):
            # 读取图像，路径为orl_faces/s1/1.bmp
            img = cv2.imread('orl_faces/s' + str(i + 1) + '/' + str(j + 1) + '.bmp', cv2.IMREAD_GRAYSCALE)
            data.append(img.flatten())
            label.append(i)

    return np.array(data), np.array(label)


# PCA降维
def pca_reduce(data, n_components):
    pca = PCA(n_components=n_components)  # 降维后的维数
    pca.fit(data)  # 训练PCA模型
    return pca.transform(data)  # 返回降维后的数据


# KNN分类
def knn_classify(train_data, train_label, test_data, k):
    knn = KNeighborsClassifier(n_neighbors=k)  # KNN分类器
    knn.fit(train_data, train_label)  # 训练KNN分类器
    return knn.predict(test_data)  # 返回预测标签


# 主函数
if __name__ == '__main__':
    # 加载数据集
    data, label = load_dataset()
    file = open('result1.txt', 'w')
    # print('Data shape:', data.shape)
    # print('Label shape:', label.shape)

    # PCA维度从10-200，knn参数从1-10，分别计算准确率
    for n_components in range(10, 201, 10):
        for k in range(1, 11):
            # PCA降维
            new_data = pca_reduce(data, n_components)

            # 使用5折交叉验证
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, new_data, label, cv=5)
            accuracy = np.mean(scores)

            print(f'PCA维度：{n_components}，KNN参数：{k}，准确率：{accuracy:.3f}', file=file)

            # 划分训练集和测试集 70%训练 30%测试
            # train_data = np.concatenate([new_data[i:i + 7, :] for i in range(0, 400, 10)])
            # train_label = np.concatenate([label[i:i + 7] for i in range(0, 400, 10)])
            # test_data = np.concatenate([new_data[i + 7:i + 10, :] for i in range(0, 400, 10)])
            # test_label = np.concatenate([label[i + 7:i + 10] for i in range(0, 400, 10)])
            # train_data, test_data, train_label, test_label = train_test_split(new_data, label, test_size=0.3, random_state=0)
            #
            # # KNN分类
            # pred_label = knn_classify(train_data, train_label, test_data, k=k)
            #
            # # 计算准确率
            # accuracy = accuracy_score(test_label, pred_label)
            # print(f'PCA维度：{n_components}，KNN参数：{k}，准确率：{accuracy:.3f}', file=file)
