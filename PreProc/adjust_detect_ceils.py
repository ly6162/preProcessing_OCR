import cv2
import numpy as np
# import table
from PIL import Image
import math
import warnings

line_detector = cv2.ximgproc.createFastLineDetector(150)

def remove_vertical_lines(lines):
    warnings.simplefilter('ignore', category=RuntimeWarning)
    threshold = 45
    l = lines.reshape((-1, 4))
    min1=l[:, 3] - l[:, 1]
    min2= l[:, 2] - l[:, 0]
    min=min1/min2
    arctan=np.arctan(min)
    degs = np.degrees(arctan) % 360
    degs = np.abs(np.where(degs > 180, degs - 360, degs))
    result = l[degs < threshold]
    return result

#横線を検出
def detect_row_lines(img):
    #グレー化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../proc_imgs/1_2_gray.jpg", gray)
    # thres, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    #適応的しきい値処理
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    cv2.imwrite("../proc_imgs/1_3_th.jpg", th)
    #元画像同じサイズの白いキャンバスを生成
    white = np.ones_like(img, dtype=np.uint8) * 255
    cv2.imwrite("../proc_imgs/1_4_white.jpg", white)
    #
    lines = line_detector.detect(th)
    result = line_detector.drawSegments(white, lines)
    cv2.imwrite("../proc_imgs/1_5_lines.jpg", result)
    #縦線を削除
    lines = remove_vertical_lines(lines)

    #lines4draw = lines.reshape(-1, 1, 4)
    #キャンバスに線を描く
    result = line_detector.drawSegments(white, lines)
    cv2.imwrite("../proc_imgs/1_6_horizontal_line.jpg", result)
    # グレー化
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../proc_imgs/1_4_gray_horizontal_line.jpg", result)

    #th1 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    #cv2.imwrite("./proc_imgs/31_th.jpg", th1)
    return lines

def cal_degrees(rows):
    if not isinstance(rows, np.ndarray):
        rows = np.array(rows)
    atan = (rows[:, 3] - rows[:, 1]) / (rows[:, 2] - rows[:, 0])
    degrees = np.degrees(np.arctan(atan)) % 360
    degrees = np.where(degrees > 180, degrees - 360, degrees)
    return np.median(degrees)

def adjust_degrees(img, degrees):
    #画像の中心サイズを計算
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    #回転(回転する中心,回転角度,拡大縮小率)
    rm = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
    #画像にアフィン変換行列を適用する。
    return cv2.warpAffine(img, rm, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))


def adjust_image(img):
    #横線を検出
    rows = detect_row_lines(img)
    #斜め度数を計算
    degrees = cal_degrees(rows)
    #回転
    warped = adjust_degrees(img, degrees)

    cv2.imwrite("../proc_imgs/1_5_warped.jpg", warped)

    return warped



if __name__ == '__main__':
    img = cv2.imdecode(np.fromfile('C:/Users/hasee/Desktop/model3/107.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    img=cv2.resize(img,(1575,2230))
    # ceils_img = gen_ceils_img(img)
    # cv2.imshow('img',ceils_img)
    warped = adjust_image(img)
    cv2.imshow('img',warped)
    cv2.waitKey(0)