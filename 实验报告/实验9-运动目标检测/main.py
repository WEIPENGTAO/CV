import cv2
import numpy as np


def non_max_suppression(boxes: np.ndarray, overlapThresh: float):
    """
    非极大值抑制
    :param boxes: 框的坐标
    :param overlapThresh: 重叠面积的阈值
    :return: 选择框的坐标
    """
    if len(boxes) == 0:
        return []

    # 如果框的类型是整数，将其转换为浮点数
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # 初始化选择框的列表
    pick = []

    # 获取框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算框的面积和排序
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # 开始循环，直到idxs中的框全部被检查
    while len(idxs) > 0:
        # 获取idxs中最后一个框的索引
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 找到重叠面积最大的框
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        # 从idxs中删除被抑制的框
        idxs = np.delete(idxs, suppress)

    # 返回选择框的坐标
    return boxes[pick].astype("int")


cap = cv2.VideoCapture('test.avi')

# 初始化背景模板
# 可以使用中值或者均值建模
# 这里使用中值建模
bg = None
accumulated_weight = 0.4

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps)

# 获取视频帧的尺寸
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

while True:
    ret, frame = cap.read()
    if ret:
        # 对图像进行预处理，转换为灰度图像并进行高斯模糊
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 如果背景为空，初始化背景
        if bg is None:
            bg = gray.copy().astype("float")
            continue

        # 更新背景模板
        cv2.accumulateWeighted(gray, bg, accumulated_weight)

        # 计算差异图
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(bg))

        # 对差异图进行二值化处理
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        # 对图像进行去噪
        thresh = cv2.medianBlur(thresh, 5)

        # 找到轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制候选框
        boxes = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 20 and h > 20:
                boxes.append([x, y, x + w, y + h])
        boxes = np.array(boxes)
        pick = non_max_suppression(boxes, overlapThresh=0.3)
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("frame", frame)
        out.write(frame)  # 将当前帧写入视频

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break

# 释放资源
cap.release()
out.release()  # 释放视频写入对象
cv2.destroyAllWindows()
