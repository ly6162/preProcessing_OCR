import cv2
import numpy as np
from PIL import Image
from PreProc import adjust_detect_ceils

def FindContours(img_path):
    npd=np.fromfile(img_path, dtype=np.uint8)
    src_img = cv2.imdecode(npd,cv2.IMREAD_COLOR)
    cv2.imwrite("../proc_imgs/1_1_org.jpg", src_img)
    #cv2.imshow("src image", src_img)

    print(img_path,'image path')
    # src_img=houghline.corect(img_path)
    #回転
    src_img= adjust_detect_ceils.adjust_image(src_img)
    cv2.imwrite("../proc_imgs/1_6_new_org.jpg", src_img)
    h, w,_= src_img.shape
    #枠線拡大(白い化)　ここ20に固定していますが、動的に計算する必要です
    #上
    src_img[0:20, :,:] = 255
    cv2.imwrite("../proc_imgs/1_7_1_new_org.jpg", src_img)
    #下
    src_img[h - 20:h, :,:] = 255
    #左
    src_img[:, 0:20,:] = 255
    cv2.imwrite("../proc_imgs/1_7_3_new_org.jpg", src_img)
    #右
    src_img[:, w - 20:w,:]=255
    cv2.imwrite("../proc_imgs/1_7_4_new_org.jpg", src_img)
    #グレー化
    src_img0 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    #ガウスぼかし
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    src_img0 = cv2.GaussianBlur(src_img0,(3,3),0)
    cv2.imwrite("../proc_imgs/1_8_GaussianBlur.jpg", src_img)
    #黒白反転
    src_img1 = cv2.bitwise_not(src_img0)
    cv2.imwrite("../proc_imgs/1_9_bitwise_not.jpg", src_img1)
    # 適応的しきい値処理
    AdaptiveThreshold = cv2.adaptiveThreshold(src_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    cv2.imwrite("../proc_imgs/1_10_th.jpg", AdaptiveThreshold)
    # thres, AdaptiveThreshold = cv2.threshold(src_img1, 150, 255, cv2.THRESH_BINARY)
    #画像コピー
    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    scale = 20
    #横線を抽出
    horizontalSize = int(horizontal.shape[1]/scale)
    #//核的形状  0：矩形  1：十字交叉形  2： 椭圆
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    # 使用菱形腐蚀图像
    horizontal = cv2.erode(horizontal, horizontalStructure)
    # 使用X膨胀原图像
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 5))
    #収縮の後に膨張 をする処理
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_GRADIENT, horizontalStructure2, (-1, -1))

    cv2.imwrite("../proc_imgs/1_11_horizontal.jpg", horizontal)
    #縦線を同じ処理
    verticalsize = int(vertical.shape[1]/scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    cv2.imwrite("../proc_imgs/1_11_vertical.jpg", vertical)
    #縦線+横線
    mask = horizontal + vertical
    cv2.imwrite("../proc_imgs/1_12_mask.jpg", mask)
    # Net_img = cv2.bitwise_and(horizontal, vertical)
    #輪郭検出
    #https://axa.biopapyrus.jp/ia/opencv/detect-contours.html
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours),'轮廓数')
    return src_img,contours

def get_Affine_Location(src_img,contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    h,w,_=src_img.shape
    #元画像と同じサイズの黒い画像せお生成
    img=np.zeros((h,w))
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    vcount=0
    hcount=0
    #横と縦の輪郭の数を数える
    for i in range(len(contours)):
        #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
        #輪郭の面積を計算する
        area0 = cv2.contourArea(contours[i])
        if area0<20 and area0==h*w :
            continue
        # =======================查找每个表的关节数====================
        #囲む周囲長
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        #輪郭の近似¶
        approx = cv2.approxPolyDP(contours[i], epsilon, True)  # 获取近似轮廓
        #外接矩形¶
        x1, y1, w1, h1 = cv2.boundingRect(approx)

        if np.sum(img[int(y1):int(y1+h1) ,int(x1):int(x1+w1)])<0.1*w1*h1*255 and h1>5 and w1>5:
            img[int(y1):int(y1 + h1), int(x1):int(x1 + w1)]=255
            if w1>h1 and h1>10:
              hcount=hcount+1
            if w1<h1 and w1>10:
              vcount = vcount + 1
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #print(vcount,hcount,'横竖转换')
    cv2.imwrite("../proc_imgs/1_14_v_h_count.jpg", img)
    return vcount,hcount

def HVDetect(input_Path):
    #直した画像(画像をまっすぐにして、色など変更なし)、輪郭検出
    src_img, contours = FindContours(input_Path)
    cv2.imwrite("./proc_imgs/1_13_src_img.jpg", src_img)
    #横と縦の輪郭の数を数えて、もし、縦の輪郭の数多いと、横した画像に判断して、まっすぐに直す
    vertical, Horizontal = get_Affine_Location(src_img, contours)
    # print(vertical, Horizontal)
    if vertical > Horizontal:
        # print("这张图片是横的")
        im=Image.fromarray(src_img, mode='RGB')
        out = im.transpose(Image.ROTATE_270)
        out=np.array(out).astype(np.uint8)
        return out
    else:
        return src_img

# if __name__ == '__main__':
#     input_Path = 'data/No2.jpg'
#     src_img, contours = FindContours(input_Path)
#     vertical,Horizontal=get_Affine_Location(src_img,contours)
#     if vertical>Horizontal:
#         print(False)
#     else:
#         print(True)
#     print(vertical,Horizontal)