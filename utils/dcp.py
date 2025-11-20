"""https://github.com/He-Zhang/image_dehaze"""
import os
import cv2
import PIL
from PIL import Image, ImageOps
import numpy as np


# opencv crop ----------------------------------------------
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
oriImage = None
rois = []


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            rois.append(roi)
            cv2.imshow("Cropped", roi)


# Methods ---------------------------------------------------
def SaturationMap(im):
    img_hsv = im.convert("HSV")
    h, s, v = img_hsv.split()  # 分离三通道
    return s


def CAPMap(im):
    img_hsv = im.convert("HSV")
    h, s, v = img_hsv.split()  # 分离三通道
    return np.asarray(v) - np.asarray(s)


def DarkChannel(im, win=15):
    """ 求暗通道图
    :param im: 输入 3 通道图
    :param win: 图像腐蚀窗口, [win x win]
    :return: 暗通道图
    """
    if isinstance(im, PIL.Image.Image):
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))  # 结构元素
    dark = cv2.erode(dc, kernel)                                    # 腐蚀操作
    return Image.fromarray(dark, mode='L')


def ShowImage(img, title='Image'):
    cv2.namedWindow(title)
    h, w, c = img.shape
    resized = cv2.resize(img, (int(w / 2), int(h / 2)))
    cv2.imshow(title, resized)  # 不 resize 的话图像太大显示不全
    cv2.waitKey(0)


def cropImage(im, dataset='Feng'):
    """'Feng', 'gz', 'Heshan', 'kaggle'"""
    w, h = im.size
    box = (0, 0, w, h)

    if 'Feng' == dataset:
        box = (w // 3, h // 2.5, w, h - h // 5)  # [top, left, down, right]
    elif 'Heshan' == dataset:
        box = (w // 3, h // 1.5, w - w // 3, h)  # [top, left, down, right]
    elif 'gz' == dataset:
        box = (0, h // 4, w - w // 2, h - h // 4)  # [top, left, down, right]

    im = im.crop(box)
    return im


if __name__ == '__main__':
    DATA_ROOT = r'D:\workplace\620资料\开题毕业\PM-小论文\science of the total environment\图例\imgs\BJNew'
    PA_DIR, CUR_DIR = DATA_ROOT.rsplit('\\', 1)
    OUTPUT_DIR = os.path.join(PA_DIR, CUR_DIR + '_output')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    for filename in os.listdir(DATA_ROOT):
        img_path = os.path.join(DATA_ROOT, filename)
        img = Image.open(img_path)  # [H, W, C]
        print(filename)

        w, h = img.size
        img = img.resize((int(w / 1), int(h / 1)))  # 缩小尺寸；鹤山 1/4，其他 1/2
        # img = cropImage(img, CUR_DIR)

        # Mouse Crop --------------------------------------------------
        oriImage = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("image", oriImage.copy())  # TODO: 两个库有点混乱
        cv2.waitKey(0)
        img = Image.fromarray(cv2.cvtColor(rois[-1], cv2.COLOR_BGR2RGB))
        # cv2.destroyAllWindows()

        dc = DarkChannel(img, win=10)
        sm = SaturationMap(img)
        ism = ImageOps.invert(sm)

        img.save(os.path.join(OUTPUT_DIR, filename))
        dc.save(os.path.join(OUTPUT_DIR, 'dc_' + filename))
        sm.save(os.path.join(OUTPUT_DIR, 'sm_' + filename))
        ism.save(os.path.join(OUTPUT_DIR, 'ism_' + filename))

    print('End...')