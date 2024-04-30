# moving object detection using two frame differencing method
import cv2
import numpy as np
from Functions import box_select
from Functions import enlargebox
from Functions import Mynet_infer
from Functions import motion_compensate
from Functions import motion_compensate_local
from Functions import cal_center_distance


def MOD2_global(frame1, frame2):
    width = frame1.shape[1]
    height = frame1.shape[0]
    blur_kernel = 11
    prveFrame = cv2.GaussianBlur(frame1, (blur_kernel, blur_kernel), 0)  # 高斯模糊，用于去噪
    prveFrame = cv2.cvtColor(prveFrame, cv2.COLOR_BGR2GRAY)  # 灰度化

    currentFame = cv2.GaussianBlur(frame2, (blur_kernel, blur_kernel), 0)
    currentFrame = cv2.cvtColor(currentFame, cv2.COLOR_BGR2GRAY)

    img_compensate, mask, avg_dist = motion_compensate(prveFrame, currentFrame)

    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(currentFrame, img_compensate)
    fix_coef = np.mean(frameDiff)
    fix_coef = int(fix_coef)
    # fix_dist = int(avg_dist * 0.1)
    fix_dist = 0
    T_1 = 5 + fix_coef + fix_dist
    retVal, thresh = cv2.threshold(frameDiff, T_1, 255, cv2.THRESH_BINARY)
    thresh1 = thresh - mask
    thresh1 = cv2.medianBlur(thresh1, 5)

    # 对阈值图像进行开操作，减少噪声
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_demo = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1, iterations=1)

    # 对开操作之后的图像做闭操作，减少孔洞，填充空隙
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_demo = cv2.morphologyEx(open_demo, cv2.MORPH_CLOSE, kernel2, iterations=3)

    # 寻找目标轮廓
    contours, hierarchy = cv2.findContours(close_demo.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # images, contours, hierarchy, for MovingDrone conda envs
    rect_list = []
    for contour in contours:
        # if contour is too small or too big, ignore it
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        ratio = w / h
        if 30 < area < 3000 and 0.6 < ratio < 3:
            rect = (x, y, w, h)
            rect_list.append(rect)

    # rect_merge = box_select(np.array(rect_list))
    rect_merge = rect_list

    if len(rect_merge) > 50:
        print('too much bboxes')
        rect_final = []
        return rect_final

    # motion classifier
    a = 2
    # rect_motion = []
    rect_final = []
    rect_candidate = []
    for i in range(len(rect_merge)):
        if np.max(rect_merge[i]) == 0:
            continue

        x0, y0, w0, h0 = rect_merge[i]

        x1, y1, w1, h1 = enlargebox(x0, y0, w0, h0, a, width, height)
        ratio1 = w1 / h1
        if ratio1 < 0.6 or ratio1 > 3:
            # print('weird bbox')
            continue

        MOD_crop1 = currentFrame[y1:y1 + h1, x1:x1 + w1]
        MOD_crop2 = img_compensate[y1:y1 + h1, x1:x1 + w1]

        # ShiTomasi corner detection的参数
        feature_params = dict(maxCorners=30, qualityLevel=0.15, minDistance=3, blockSize=3)
        lk_params = dict(winSize=(15, 15), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

        pts1 = cv2.goodFeaturesToTrack(MOD_crop1, mask=None, **feature_params)

        if pts1 is None:
            # rect_motion0 = (x1, y1, w1, h1)
            # rect_motion.append(rect_motion0)
            # print('no feature points')
            continue

        pts2, st1, err1 = cv2.calcOpticalFlowPyrLK(MOD_crop2, MOD_crop1, pts1, None, **lk_params)
        good_new1 = pts2[st1 == 1]
        good_old1 = pts1[st1 == 1]

        if len(good_new1) < 1:
            # print("few feature points")
            continue

        motion_dist = []
        motion_theta = []
        for j, (new, old) in enumerate(zip(good_new1, good_old1)):
            a1, b1 = new.ravel()
            c1, d1 = old.ravel()
            motion_distance0 = np.sqrt((a1 - c1) * (a1 - c1) + (b1 - d1) * (b1 - d1))
            motion_theta0 = 57.3 * np.arctan2(d1 - b1, c1 - a1)
            motion_dist.append(motion_distance0)
            motion_theta.append(motion_theta0)

        motion_dist = np.array(motion_dist)
        motion_theta = np.array(motion_theta)

        std_theta = np.std(motion_theta)
        avg_theta = np.mean(motion_theta)
        ratio_theta = std_theta / avg_theta

        std_dist = np.std(motion_dist)
        avg_dist = np.mean(motion_dist)
        ratio_dist = std_dist / avg_dist

        if avg_dist < 1 or ratio_theta > 0.8 or ratio_dist > 0.8:
            # print('erratic motion box removed')
            continue

        MOD_crop = frame1[y1:y1 + h1, x1:x1 + w1, :]
        index = Mynet_infer(MOD_crop)
        if index == 1:
            rect_final = (x1, y1, w1, h1)
            # rect_candidate0 = [x1, y1, w1, h1]
            # rect_candidate.append(rect_candidate0)
            break

    return rect_final


def MOD2_local(frame1, frame2, x_prev, y_prev):
    width = frame1.shape[1]
    height = frame1.shape[0]
    blur_kernel = 11
    prveFrame = cv2.GaussianBlur(frame1, (blur_kernel, blur_kernel), 0)  # 高斯模糊，用于去噪
    prveFrame = cv2.cvtColor(prveFrame, cv2.COLOR_BGR2GRAY)  # 灰度化

    currentFame = cv2.GaussianBlur(frame2, (blur_kernel, blur_kernel), 0)
    currentFrame = cv2.cvtColor(currentFame, cv2.COLOR_BGR2GRAY)

    img_compensate, mask, homo_inv = motion_compensate_local(prveFrame, currentFrame)

    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(currentFrame, img_compensate)
    retVal, thresh = cv2.threshold(frameDiff, 4, 255, cv2.THRESH_BINARY)
    thresh1 = thresh - mask

    # 对阈值图像进行开操作，减少噪声
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_demo = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1, iterations=1)

    # 对开操作之后的图像做闭操作，减少孔洞，填充空隙
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_demo = cv2.morphologyEx(open_demo, cv2.MORPH_CLOSE, kernel2, iterations=3)

    # 寻找目标轮廓
    contours, hierarchy = cv2.findContours(close_demo.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # images, contours, hierarchy, for MovingDrone conda envs
    rect_list = []
    for contour in contours:
        # if contour is too small or too big, ignore it
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        ratio = w / h
        if 30 < area < 3000 and 0.5 < ratio < 3:
            rect = (x, y, w, h)
            rect_list.append(rect)

        # rect_merge = box_select(np.array(rect_list))
    rect_merge = rect_list

    if len(rect_merge) > 30:
        print('too much bboxes')
        rect_final = []
        return rect_final

    # motion classifier
    rect_final = []
    a = 2
    rect_candidate = []
    dist_ref = 200
    for i in range(len(rect_merge)):
        if np.max(rect_merge[i]) == 0:
            continue

        x0, y0, w0, h0 = rect_merge[i]

        x1, y1, w1, h1 = enlargebox(x0, y0, w0, h0, a, width, height)
        ratio1 = w1 / h1
        if ratio1 < 0.6 or ratio1 > 3:
            # print('weird bbox')
            continue

        MOD_crop1 = currentFrame[y1:y1 + h1, x1:x1 + w1]
        MOD_crop2 = img_compensate[y1:y1 + h1, x1:x1 + w1]

        # ShiTomasi corner detection的参数
        feature_params = dict(maxCorners=30, qualityLevel=0.15, minDistance=3, blockSize=3)
        lk_params = dict(winSize=(15, 15), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

        pts1 = cv2.goodFeaturesToTrack(MOD_crop1, mask=None, **feature_params)

        if pts1 is None:
            # rect_motion0 = (x1, y1, w1, h1)
            # rect_motion.append(rect_motion0)
            # print('no feature points')
            continue

        pts2, st1, err1 = cv2.calcOpticalFlowPyrLK(MOD_crop2, MOD_crop1, pts1, None, **lk_params)
        good_new1 = pts2[st1 == 1]
        good_old1 = pts1[st1 == 1]

        if len(good_new1) < 1:
            # print("few feature points")
            continue

        motion_dist = []
        motion_theta = []
        for j, (new, old) in enumerate(zip(good_new1, good_old1)):
            a1, b1 = new.ravel()
            c1, d1 = old.ravel()
            motion_distance0 = np.sqrt((a1 - c1) * (a1 - c1) + (b1 - d1) * (b1 - d1))
            motion_theta0 = 57.3 * np.arctan2(d1 - b1, c1 - a1)
            motion_dist.append(motion_distance0)
            motion_theta.append(motion_theta0)

        motion_dist = np.array(motion_dist)
        motion_theta = np.array(motion_theta)

        std_theta = np.std(motion_theta)
        avg_theta = np.mean(motion_theta)
        ratio_theta = std_theta / avg_theta

        std_dist = np.std(motion_dist)
        avg_dist = np.mean(motion_dist)
        ratio_dist = std_dist / avg_dist

        if avg_dist < 0.6 or ratio_theta > 1 or ratio_dist > 1:
            # print('erratic motion box removed')
            continue

        MOD_crop = frame1[y1:y1 + h1, x1:x1 + w1, :]
        index = Mynet_infer(MOD_crop)
        if index == 1:
            rect_candidate0 = [x1, y1, w1, h1]
            rect_candidate.append(rect_candidate0)
            x_now = x1 + w1 / 2
            y_now = y1 + h1 / 2
            dist = np.sqrt((x_now - x_prev) * (x_now - x_prev) + (y_now - y_prev) * (y_now - y_prev))
            if dist < dist_ref:
                rect_final = (x1, y1, w1, h1)
                dist_ref = dist

    return rect_final



