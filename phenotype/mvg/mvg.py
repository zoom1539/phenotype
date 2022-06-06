import numpy as np
import cv2
from math import sqrt


def triangulate1(K1, K2, T_2_1, point_img1, point_img2):
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

def triangulate2(Ks, Ts, coords):
    """ 
    :param Ks: (n, 3,3)
    :param Ts: (n, 4,4) world to cam
    :param coords: (n, 3), (x,y,score)
    :returns: (n, 4), (x,y,z,score)
    """
    A = []
    for K, T, coord in zip(Ks, Ts, coords):
        P = K.dot(np.linalg.inv(T)[:3, :4])
        
        a = coord[0] * P[2] - P[0]
        a = coord[2] * a / np.linalg.norm(a)
        A.append(a)
        
        a = coord[1] * P[2] - P[1]
        a = coord[2] * a / np.linalg.norm(a)
        A.append(a)
    
    A = np.array(A)
    
    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]
    point_3d = point_3d_homo.T[:-1] / point_3d_homo.T[-1]

    return np.append(point_3d, 1)

def calc_epipolar_dist(kp1, K1, T1, kp2, K2, T2):
    """ refer to 视觉slam 十四讲
    :param kp1: (3,)(x,y,score)
    :param K1: (3,3)
    :param T1: (4,4)
    :param kp2: (3,)(x,y,score)
    :param K2: (3,3)
    :param T2: (4,4)
    :returns: epipolar dist
    """
    T_2_1 = np.linalg.inv(T2).dot(T1)
    R = T_2_1[:3, :3]
    t = T_2_1[:3, 3]
    s = np.array([[0, -t[2], t[1]],
                  [t[2],0,  -t[0]],
                  [-t[1],t[0], 0]])
    E = np.dot(s, R)

    F = np.linalg.inv(K2).T.dot(E).dot(np.linalg.inv(K1)) 

    p1 = np.append(kp1[:2], 1)
    p2 = np.append(kp2[:2], 1)
    line = F.dot(p1)
    norm = sqrt(line[0]**2 + line[1]**2)
    dist1 = abs(p2.T.dot(line)) / norm

    line = F.T.dot(p2)
    norm = sqrt(line[0]**2 + line[1]**2)
    dist2 = abs(p1.T.dot(line)) / norm

    return (dist1 + dist2) * 0.5



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
#----
    R = T_2_1[:3, :3]
    t = T_2_1[:3, 3]
    s = np.array([[0, -t[2], t[1]],
                  [t[2],0,  -t[0]],
                  [-t[1],t[0], 0]])
    
    E = np.dot(s, R)

    F = np.linalg.inv(matrix2).T.dot(E).dot(np.linalg.inv(matrix1))

    p1 = np.append(corners1[1], 1)
    p2 = np.append(corners2[0], 1)
    print(p1)
    print(p2)
    n = F.dot(p1)
    norm = sqrt(n[0]**2 + n[1]**2)
    print(p2.T.dot(F).dot(p1) / norm)

    pts1 = []
    pts1.append(p1)
    pts1 = np.array(pts1)

    epilines_1to2 = np.squeeze(
    cv2.computeCorrespondEpilines(pts1, 1, F))
    print(epilines_1to2)
    a = epilines_1to2[0]
    b = epilines_1to2[1]
    c = epilines_1to2[2]
    lp1 = (0, int(-c / b))
    lp2 = (3072, int((-c - a*3072) / b))
    cv2.line(img, lp1, lp2, (0, 255,0), 3)




    for i in np.arange(8):
        print('-------------', i)
        for j in np.arange(11):
            p2 = np.append(corners2[i * 11 + j], 1)
            # el = p1.T.dot(F)
            # print(el.dot(p2))
            print(abs(a * p2[0] + b * p2[1] + c) / sqrt(a**2 + b**2))


    img = cv2.resize(img, (1200, 800))
    cv2.imshow('findCorners',img)
    cv2.waitKey()

    img1 = cv2.imread(img1)
    p2 = np.append(corners2[0], 1)
    n = F.T.dot(p2)
    a = n[0]
    b = n[1]
    c = n[2]
    lp1 = (0, int(-c / b))
    lp2 = (3072, int((-c - a*3072) / b))
    cv2.line(img1, lp1, lp2, (0, 255,0), 3)

    norm = sqrt(n[0]**2 + n[1]**2)
    p1 = np.append(corners1[1], 1)
    print(p1.T.dot(F.T).dot(p2) / norm)

    img1 = cv2.resize(img1, (1200, 800))
    cv2.imshow('img1',img1)
    cv2.waitKey()

