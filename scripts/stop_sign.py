
import cv2
import numpy as np
import os

# 读取图片
for file in os.listdir('/Users/zhangwenbo/Documents/RVSS_Need4Speed/stop_sign_big'):
    image = cv2.imread('/Users/zhangwenbo/Documents/RVSS_Need4Speed/stop_sign_big/'+file)

    # 将图片转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的两个HSV范围
    lower_red1 = np.array([0, 160, 140])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([170, 160, 140])
    upper_red2 = np.array([180, 255, 255])

    # 根据定义的颜色范围创建两个掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合并掩码
    mask = cv2.bitwise_or(mask1, mask2)

    # 将掩码和原图进行位运算，只显示红色区域
    result = cv2.bitwise_and(image, image, mask=mask)
    print(np.sum(mask))
    result = cv2.imwrite('/Users/zhangwenbo/Documents/RVSS_Need4Speed/stop_sign_seg/' + file.replace('stop_sign','stop_sign_seg'), mask)


# 显示结果
# cv2.imshow('Original', image)
# cv2.imshow('Mask', mask)
# cv2.imshow('Detected Red Area', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
