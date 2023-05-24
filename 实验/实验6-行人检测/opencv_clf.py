import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# from sklearn.metrics import roc_curve, auc


def load_images(dir_path: str) -> list:
    """
    从文件中读取图片。
    :param dir_path:
    :return: img_list: 图片列表
    """
    img_list = []
    for img_name in os.listdir(dir_path):
        img = Image.open(os.path.join(dir_path, img_name)).convert('RGB')
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # 转成 OpenCV 格式
        img_list.append(img)
    return img_list


def sample_neg(full_neg_lst: list, size: tuple, num_samples: int = 10) -> list:
    """
    从每一张没有人的原始图片中随机裁出指定数量的小图片作为负样本。

    :param full_neg_lst: 包含所有原始图片的列表
    :param size: 要裁剪的小图片的大小 (height, width)
    :param num_samples: 每张原始图片要裁剪的小图片数量，默认为 10
    :return: 由裁剪后的小图片组成的列表
    """
    neg_list = []
    height, width = size
    rng = np.random.default_rng(1)

    for img in full_neg_lst:
        for _ in range(num_samples):
            y, x = rng.integers(img.shape[0] - height), rng.integers(img.shape[1] - width)  # 随机裁剪
            neg_list.append(img[y:y + height, x:x + width])  # 添加到列表中

            if len(neg_list) >= len(full_neg_lst) * num_samples:  # 裁剪的小图片数量达到上限
                return neg_list

    return neg_list


# window_size: 处理图片大小，通常64*128; 输入图片尺寸>= window_size
def compute_HOGs(img_lst: list, window_size: tuple = (128, 64)):
    """
    计算一组图像的 HOG 特征。
    :param img_lst: 输入图像列表。
    :param window_size: 窗口大小。
    """
    hog = cv2.HOGDescriptor()
    gradient_lst = []

    for i in range(len(img_lst)):
        if img_lst[i].shape[1] >= window_size[1] and img_lst[i].shape[0] >= window_size[0]:
            # 在原始图像中截取指定大小的窗口
            roi = img_lst[i][(img_lst[i].shape[0] - window_size[0]) // 2: (img_lst[i].shape[0] - window_size[0]) // 2 +
                                                                          window_size[0], \
                  (img_lst[i].shape[1] - window_size[1]) // 2: (img_lst[i].shape[1] - window_size[1]) // 2 +
                                                               window_size[1]]

            # 转换为灰度图像
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 计算 HOG 特征
            gradient_lst.append(hog.compute(gray))

    # 返回梯度列表
    return gradient_lst


def get_svm_detector(svm: cv2.ml_SVM):
    """
    获取 SVM 分类器的检测器（即分类超平面）。
    :param svm: 输入的 SVM 分类器。
    :return: 返回分类器的检测器。
    """
    # 获取支持向量和决策函数参数
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)

    # 转置支持向量矩阵并添加偏差
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, save_path: str):
    """
    绘制 ROC 曲线并保存
    :param fpr: 假正例率
    :param tpr: 真正例率
    :param auc_score: AUC 值
    :param save_path: 保存路径
    :return: None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()


# 主程序
if __name__ == '__main__':
    # 主程序
    # 第一步：计算HOG特征
    neg_list = []
    pos_list = []
    labels = []
    hard_neg_list = []

    # 加载正样本和负样本的图像
    pos_list = load_images(
        r'E:/wpt/Documents/CS/计算机视觉/实验报告/实验6-行人检测/INRIADATA/normalized_images/train/pos')
    full_neg_lst = load_images(
        r'E:/wpt/Documents/CS/计算机视觉/实验报告/实验6-行人检测/INRIADATA/normalized_images/train/neg')

    # 对负样本进行采样
    print('Sample negative features')
    neg_list = sample_neg(full_neg_lst, [128, 64])

    # 计算正样本的 HOG 特征，并为每个样本添加标签
    print('Compute positive features')
    pos_gradients = compute_HOGs(pos_list)
    labels += [+1] * len(pos_list)

    # 计算负样本的 HOG 特征，并为每个样本添加标签
    print('Compute negative features')
    neg_gradients = compute_HOGs(neg_list)
    labels += [-1] * len(neg_list)

    # 将正样本和负样本的 HOG 特征合并
    gradient_lst = pos_gradients + neg_gradients

    # 第二步：训练SVM
    svm = cv2.ml.SVM_create()

    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    # 用计算得到的HOG特征和对应的标签训练SVM分类器
    print('Train SVM ...')
    svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

    # 第三步：加入识别错误的样本，进行第二轮训练
    hog = cv2.HOGDescriptor()
    hard_neg_list.clear()
    hog.setSVMDetector(get_svm_detector(svm))
    for i in range(len(full_neg_lst)):
        rects, wei = hog.detectMultiScale(full_neg_lst[i], winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            hardExample = full_neg_lst[i][y:y + h, x:x + w]
            hard_neg_list.append(cv2.resize(hardExample, (64, 128)))
    hard_neg_gradients = compute_HOGs(hard_neg_list)
    gradient_lst += hard_neg_gradients
    [labels.append(-1) for _ in range(len(hard_neg_list))]
    svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))

    # # 测试 SVM 分类器
    # test_pos_list = load_images(
    #     'E:/wpt/Documents/CS/计算机视觉/实验报告/实验6-行人检测/INRIADATA/normalized_images/test/pos')
    # test_full_neg_list = load_images(
    #     'E:/wpt/Documents/CS/计算机视觉/实验报告/实验6-行人检测/INRIADATA/original_images/test/neg')
    #
    # # 对负样本进行采样
    # print('Sample negative features')
    # test_neg_list = sample_neg(test_full_neg_list, [128, 64])
    #
    # # 计算测试集的 HOG 特征
    # print('Compute positive features')
    # test_pos_gradient_lst = compute_HOGs(test_pos_list)
    # test_labels = [+1] * len(test_pos_list)
    #
    # print('Compute negative features')
    # test_neg_gradient_lst = compute_HOGs(test_neg_list)
    # test_labels += [-1] * len(test_neg_list)
    #
    # test_gradient_lst = test_pos_gradient_lst + test_neg_gradient_lst
    #
    # # 对测试集进行预测
    # result = svm.predict(np.array(test_gradient_lst))[1].ravel()
    #
    # # 计算准确率
    # mask = result == test_labels
    # correct = np.count_nonzero(mask)
    # print('Accuracy: %.2f %%' % (correct * 100.0 / result.size))
    #
    # # 绘制 ROC 曲线
    # fpr, tpr, _ = roc_curve(test_labels, result)
    # roc_auc = auc(fpr, tpr)
    # plot_roc_curve(fpr, tpr, roc_auc, 'ROC curve of SVM', 'svm_roc.png')

    # 第四步：保存训练结果
    print('Save SVM ...')
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHogDector.bin')
