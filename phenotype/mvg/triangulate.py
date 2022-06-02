#refer to https://blog.csdn.net/qq_42995327/article/details/118917141

import numpy as np
import cv2
def triangulte(K1, K2, T_2_1, point_img1, point_img2):
    T1 = np.zeros((3,4))
    T1[:3,:3] = np.eye(3,3)
    P1 = K1.dot(T1)

    T2 = T_2_1[:3, :4]
    P2 = K2.dot(T2)

    A = np.zeros((4,4))
    A[0] = point_img1[0]*P1[2] - P1[0]
    A[1] = point_img1[1]*P1[2] - P1[1]
    A[2] = point_img2[0]*P2[2] - P2[0]
    A[3] = point_img2[1]*P2[2] - P2[1]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]
    point_3d = point_3d_homo.T[:-1] / point_3d_homo.T[-1]

    return point_3d


if __name__ == '__main__':
    cam1 = '1'
    cam2 = '2'


    #
    folder = '/home/ubuntu/Documents/phenotype_device/data/calib/MultiCamera/' + cam1 + '_' + cam2 + '/'
    img1 = folder + cam1 + '_20220422152602.jpg'
    img2 = folder + cam2 + '_20220422152602.jpg'

    fs = cv2.FileStorage(folder + 'result.yml', cv2.FILE_STORAGE_READ)
    T_1_2 = fs.getNode('T_1_2').mat()


    calib_folder = '/home/ubuntu/Documents/phenotype_device/data/calib/OneCamera/' + cam1 + '/'
    fs = cv2.FileStorage(calib_folder + 'result.yml', cv2.FILE_STORAGE_READ)
    matrix1 = fs.getNode('matrix').mat()
    dist1 = fs.getNode('dist').mat()

    calib_folder = '/home/ubuntu/Documents/phenotype_device/data/calib/OneCamera/' + cam2 + '/'
    fs = cv2.FileStorage(calib_folder + 'result.yml', cv2.FILE_STORAGE_READ)
    matrix2 = fs.getNode('matrix').mat()
    dist2 = fs.getNode('dist').mat()

    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #
    img = cv2.imread(img1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners1 = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners1,(11,11),(-1,-1),criteria)

    #
    img = cv2.imread(img2)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners2 = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners2,(11,11),(-1,-1),criteria)
    
    corners1 = np.squeeze(corners1)
    corners2 = np.squeeze(corners2)

    T_2_1 = np.linalg.inv(T_1_2)

    point_pre = [0,0,0]
    dists = []
    for corner1, corner2 in zip(corners1, corners2):
        point_3d = triangulte(matrix1, matrix2, T_2_1, corner1, corner2)
        # print(point_3d)
        dist = np.linalg.norm(point_pre - point_3d)
        point_pre = point_3d
        if(dist < 0.1):
            dists.append(dist)
            print(dist)
    
    # mae
    print('mae')
    print(np.mean(np.abs(np.array(dists) - 0.045)))
    print(np.max(np.abs(np.array(dists) - 0.045)))
    print(np.min(np.abs(np.array(dists) - 0.045)))




    
