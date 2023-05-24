import cv2
import numpy as np

# 创建 HOG 特征提取器
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')

# 读取图片
img = cv2.imread('img1.jpg')

# 检测行人
rects, weights = hog.detectMultiScale(
    img,
    winStride=(4, 4),  # 滑动窗口步长
    padding=(8, 8),  # 填充
    scale=1.05  # 缩放因子
)

# 转换为 x1, y1, x2, y2 的形式
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

# 非极大值抑制
conf_threshold = 0.1  # 置信度阈值
nms_threshold = 0.4  # 非极大值抑制阈值
weights = np.asarray(weights)
keep = cv2.dnn.NMSBoxes(rects, weights.flatten(), conf_threshold, nms_threshold)

# 绘制检测结果, 并显示置信度
for i in keep:
    x1, y1, x2, y2 = rects[i]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(round(weights[i], 2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
