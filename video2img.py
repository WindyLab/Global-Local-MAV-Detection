import cv2

vc = cv2.VideoCapture('./videos/phantom16.mp4')
c = 0
rval = vc.isOpened()

while rval:
    c = c + 1
    rval, frame = vc.read()
    if rval:
        if c % 1 == 0:
            name0 = str(c)
            name = name0.zfill(4)
            cv2.imwrite('./images/phantom16/' + 'phantom16_' + name + '.jpg', frame)
            print('extract frame: ', name)
        else:
            continue
    else:
        break

vc.release()


import cv2
import os

# 设置视频文件夹路径
video_folder = './videos/'
image_folder = './images/'
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# 遍历所有视频文件
for video_file in video_files:
    # 创建每个视频的图像文件夹
    video_image_folder = os.path.join(image_folder, video_file.split('.')[0])
    if not os.path.exists(video_image_folder):
        os.makedirs(video_image_folder)

    # 开始处理视频文件
    vc = cv2.VideoCapture(os.path.join(video_folder, video_file))
    c = 0
    rval = vc.isOpened()

    while rval:
        c += 1
        rval, frame = vc.read()
        if rval:
            # 每帧都写入到对应的文件夹中
            name0 = str(c)
            name = name0.zfill(4)
            cv2.imwrite(os.path.join(video_image_folder, video_file.split('.')[0] + '_' + name + '.jpg'), frame)
            print(f'extract frame from {video_file}:', name)
        else:
            break

    vc.release()
