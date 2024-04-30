import cv2
import numpy as np
import time

from detector1_trt import Detector1
from detector2_trt import Detector2
import ctypes

from MOD2 import MOD2_global
from MOD2 import MOD2_local
from Functions import frame_stablize
from Functions import enlarge_region2


PLUGIN_LIBRARY = "./weights/libmyplugins.so"
ctypes.CDLL(PLUGIN_LIBRARY)
engine_file_path1 = './weights/yolov5s_DT-Drone2.engine'
engine_file_path2 = './weights/yolov5s_DT-Drone2-crop.engine'
detector1 = Detector1(engine_file_path1)
detector2 = Detector2(engine_file_path2)

sets_ordinary = ['phantom09', 'phantom10', 'phantom30', 'phantom47', 'phantom70']
sets_complex = ['phantom05', 'phantom08', 'phantom58', 'phantom65', 'phantom86']
sets_small = ['phantom19', 'phantom41', 'phantom43', 'phantom46', 'phantom63']

sets_test = ['phantom09']

border = 1

for i in range(len(sets_test)):
    video_name = sets_test[i]

    cap = cv2.VideoCapture('/home/user-guo/data/ARD-MAV/videos/' + video_name + '.mp4')

    count = 0
    flag = 0
    prveframe = None
    fail_num = 0
    a = 160

    print('read file: ', video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        # print(ret)
        if not ret:
            break

        if prveframe is None:
            print('first frame input')
            prveframe = frame
            count = count + 1
            continue

        frame_show = frame.copy()
        width = frame.shape[1]
        height = frame.shape[0]
        t1 = time.time()

        if flag == 0:
            boxes = detector1.detect(frame)
            if len(boxes) == 0:
                boxes_MOD = MOD2_global(prveframe, frame)
                if len(boxes_MOD) != 0:
                    (x, y) = (boxes_MOD[0], boxes_MOD[1])
                    (w, h) = (boxes_MOD[2], boxes_MOD[3])

                    init_rect = [x, y, w, h]
                    flag = 1
                    fail_num = 0

                    xleft = x
                    ytop = y
                    xright = x + w
                    ybottom = y + h

                    x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)

                    # 画出边框和标签
                    color = (255, 0, 0)
                    cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)

                    x2 = x - x1
                    y2 = y - y1
                    w2 = w
                    h2 = h
                else:
                    flag = 0
                    init_rect = []
                    # status = 'Both Failure'
            else:
                (x, y) = (boxes[0], boxes[1])
                (w, h) = (boxes[2], boxes[3])
                init_rect = [x, y, w, h]

                xleft = x
                ytop = y
                xright = x + w
                ybottom = y + h

                color = (0, 255, 255)
                cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                fail_num = 0
                flag = 1
                x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)

                x2 = x - x1
                y2 = y - y1
                w2 = w
                h2 = h
                # status = 'Global YOLO'
        else:
            homo_inv = frame_stablize(prveframe, frame)
            search_box = np.array([[x1, y1], [x1 + w1, y1], [x1 + w1, y1 + h1], [x1, y1 + h1]], dtype=np.float32).reshape(-1, 1, 2)
            search_box_s = cv2.perspectiveTransform(search_box, homo_inv)
            search_box_new = np.array(search_box_s, dtype=np.int32).reshape(4, 2)

            detect_box = np.array([[xleft, ytop], [xright, ytop], [xright, ybottom], [xleft, ybottom]], dtype=np.float32).reshape(-1, 1, 2)
            detect_box_s = cv2.perspectiveTransform(detect_box, homo_inv)
            detect_box_new = np.array(detect_box_s, dtype=np.int32).reshape(4, 2)

            if search_box_new[0][0] < 0:
                search_box_new[0][0] = 0

            if search_box_new[0][1] < 0:
                search_box_new[0][1] = 0

            if search_box_new[2][1] > height:
                search_box_new[2][1] = height

            if search_box_new[1][0] > width:
                search_box_new[1][0] = width

            track_crop1 = prveframe[search_box_new[0][1]:search_box_new[2][1], search_box_new[0][0]:search_box_new[1][0], :]
            track_crop2 = frame[search_box_new[0][1]:search_box_new[2][1], search_box_new[0][0]:search_box_new[1][0], :]
            x_prve = (detect_box_new[0][0] + detect_box_new[2][0] - search_box_new[0][0] * 2) / 2
            y_prve = (detect_box_new[0][1] + detect_box_new[2][1] - search_box_new[0][1] * 2) / 2
            boxes = detector2.detect(track_crop2, x_prve, y_prve)

            if len(boxes) == 0:
                boxes_MOD = MOD2_local(track_crop1, track_crop2, x_prve, y_prve)
                if len(boxes_MOD) != 0:
                    (x2, y2) = (boxes_MOD[0], boxes_MOD[1])
                    (w2, h2) = (boxes_MOD[2], boxes_MOD[3])

                    init_rect = [x2 + search_box_new[0][0], y2 + search_box_new[0][1], w2, h2]
                    xleft = x2 + search_box_new[0][0]
                    ytop = y2 + search_box_new[0][1]
                    xright = x2 + search_box_new[0][0] + w2
                    ybottom = y2 + search_box_new[0][1] + h2

                    # 画出边框和标签
                    color = (255, 0, 0)
                    cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame_show, (search_box_new[0][0], search_box_new[0][1]), (search_box_new[1][0], search_box_new[2][1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(frame_show, "search region", (search_box_new[0][0] + 20, search_box_new[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    fail_num = 0
                    flag = 2
                    x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)
                    # status = 'Local MOD'
                else:
                    fail_num = fail_num + 1
                    init_rect = []
                    # status = 'Local Both Failure'
            else:
                (x2, y2) = (boxes[0], boxes[1])
                (w2, h2) = (boxes[2], boxes[3])

                init_rect = [x2 + search_box_new[0][0], y2 + search_box_new[0][1], w2, h2]
                xleft = x2 + search_box_new[0][0]
                ytop = y2 + search_box_new[0][1]
                xright = x2 + search_box_new[0][0] + w2
                ybottom = y2 + search_box_new[0][1] + h2

                color = (0, 255, 255)
                cv2.rectangle(frame_show, (xleft, ytop), (xright, ybottom), color, border, lineType=cv2.LINE_AA)
                cv2.rectangle(frame_show, (search_box_new[0][0], search_box_new[0][1]), (search_box_new[1][0], search_box_new[2][1]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame_show, "search region", (search_box_new[0][0] + 20, search_box_new[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                fail_num = 0
                flag = 2
                x1, y1, w1, h1 = enlarge_region2(xleft, ytop, a, width, height)
            if fail_num == 30:
                print('turn to global re-detection')
                flag = 0

        print(video_name, end=" ")
        print('frame count: %d' % count, end=' ')
        print('bbox:', init_rect)
        cv2.imshow('GLAD', frame_show)
        count = count + 1
        prveframe = frame
        key = cv2.waitKey(10) & 0xff

        if key == 27 or key == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()




