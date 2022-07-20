import cv2

from PreProc import table_hv_detect, upside_down
import os
import shutil


def print_hi(path):

    path="./proc_imgs"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    img_path="./imgs/5_upside_right.png"

    img = cv2.imread(img_path)

    cv2.imwrite("./proc_imgs/3_input.jpg", img)
    #斜め、横の画像をまっすぐに直す
    new_img = table_hv_detect.HVDetect(img_path)

    cv2.imwrite("./proc_imgs/1_20_final.jpg", new_img)
    final_img= upside_down.proc(new_img)
    cv2.imwrite("./proc_imgs/2_2_final.jpg", final_img)
    cv2.imwrite("./proc_imgs/3_output.jpg", final_img)
    pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('./imgs/2 - rightHorizontal.jpg')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
