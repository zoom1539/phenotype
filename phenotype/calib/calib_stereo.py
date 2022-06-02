import cv2
import numpy as np
import glob
# refer to https://blog.csdn.net/briblue/article/details/86569448
# refer to https://blog.csdn.net/weian4913/article/details/95523678

###param
cam1 = '6'
cam2 = '7'


#
folder = '/home/ubuntu/Documents/phenotype_device/data/calib/MultiCamera/' + cam1 + '_' + cam2 + '/'
img1 = folder + cam1 + '_20220422161959.jpg'
img2 = folder + cam2 + '_20220422161959.jpg'

calib_folder = '/home/ubuntu/Documents/phenotype_device/data/calib/OneCamera/' + cam1 + '/'
fs = cv2.FileStorage(calib_folder + 'result.yml', cv2.FILE_STORAGE_READ)
matrix1 = fs.getNode('matrix').mat()
dist1 = fs.getNode('dist').mat()

calib_folder = '/home/ubuntu/Documents/phenotype_device/data/calib/OneCamera/' + cam2 + '/'
fs = cv2.FileStorage(calib_folder + 'result.yml', cv2.FILE_STORAGE_READ)
matrix2 = fs.getNode('matrix').mat()
dist2 = fs.getNode('dist').mat()

###start
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 11
h = 8
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2) * 0.045

#
def calc_ex(objp, img_path, w, h):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        img = cv2.resize(img, (1200, 800))
        cv2.imshow('findCorners',img)
        cv2.waitKey()

        (success, rotation_vector, translation_vector) = cv2.solvePnP(objp, 
                                                                    corners, 
                                                                    matrix1, 
                                                                    dist1, 
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        rotM = cv2.Rodrigues(rotation_vector)[0]

        ex = np.eye(4,4)
        ex[:3,:3] = rotM
        ex[:3, 3] = translation_vector.reshape(-1)

        return ex

ex1 = calc_ex(objp, img1, w, h)
ex2 = calc_ex(objp, img2, w, h)

T_6_7 = np.matrix(ex1) * np.matrix(ex2).I
print(ex1)
print(ex2)
print(T_6_7)

fs = cv2.FileStorage(folder + 'result.yml', cv2.FILE_STORAGE_WRITE)
fs.write('T_6_7', T_6_7)
fs.release()
