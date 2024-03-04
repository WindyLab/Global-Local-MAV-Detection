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
