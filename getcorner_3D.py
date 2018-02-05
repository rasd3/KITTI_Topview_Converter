# make 3D coordinates from train Label data

# train 7481, test 7518

import os
import pdb
import cv2
import math
import ctypes
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
        TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
import numpy as np

# open so file (lidar npy -> 500x300x15(height:13 + intensity + density))
so_path = "./LidarTopPreprocess.so"
clidarLib = ctypes.cdll.LoadLibrary(so_path)

class obj(object):
    def __init__(self, C):

        self.type       = C[0]
        self.truncation = float(C[1])
        self.occlusion  = int(C[2])
        self.alpha      = float(C[3])

        self.x1 = float(C[4])
        self.y1 = float(C[5])
        self.x2 = float(C[6])
        self.y2 = float(C[7])
        
        self.h      = float(C[8])
        self.w      = float(C[9])
        self.l      = float(C[10])
        self.t      = [float(C[11]), float(C[12]), float(C[13])]
        self.ry     = float(C[14])

        self.R = np.array([[math.cos(self.ry), 0, math.sin(self.ry)], \
                [0, 1, 0], [-math.sin(self.ry), 0, math.cos(self.ry)]])
        self.x_corners = np.array([self.l/2, self.l/2, -self.l/2, -self.l/2, self.l/2, self.l/2, -self.l/2, -self.l/2])
        self.y_corners = np.array([0, 0, 0, 0, -self.h, -self.h, -self.h, -self.h])
        self.z_corners = np.array([self.w/2, -self.w/2, -self.w/2, self.w/2, self.w/2, -self.w/2, -self.w/2, self.w/2])

        self.corners_3D = np.dot(self.R, np.concatenate(([self.x_corners], \
                [self.y_corners], [self.z_corners]), axis=0))
        self.corners_3D[0, :] += self.t[0]
        self.corners_3D[1, :] += self.t[1]
        self.corners_3D[2, :] += self.t[2]

    def showCoor(self, idx):
        print("%d cor\n" % idx)
        for i in range(8):
            print("(%f, %f, %f)\n" % (self.corners_3D[0, i], self.corners_3D[1, i], self.corners_3D[2, i]))

    def draw_bbox(self, ch3_dir, chk_dir):
        if os.path.isfile(chk_dir) == False:
            img = cv2.imread(ch3_dir)
            cv2.imwrite(chk_dir, img)

        img = cv2.imread(chk_dir)
        pts = []
        for i in range(4):
            pts.append([self.corners_3D[0, i], self.corners_3D[2, i]])

        pts = box3d_to_top_box(pts) 
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], True, (0, 0, 255))
        cv2.imwrite(chk_dir, img)

# from MV3D data.py
def clidar_to_top(lidar):
    # Calculate map size and pack parameters for top view and front view map (DON'T CHANGE THIS !)
    Xn = int((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION)
    Yn = int((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION)
    Zn = int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)

    top_flip = np.ones((Xn, Yn, Zn + 2), dtype=np.float32)  # DON'T CHANGE THIS !

    num = lidar.shape[0]  # DON'T CHANGE THIS !

    # call the C function to create top view maps
    # The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps
    clidarLib.createTopMaps(ctypes.c_void_p(lidar.ctypes.data),
                            ctypes.c_int(num),
                            ctypes.c_void_p(top_flip.ctypes.data),
                            ctypes.c_float(TOP_X_MIN), ctypes.c_float(TOP_X_MAX),
                            ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Y_MAX),
                            ctypes.c_float(TOP_Z_MIN), ctypes.c_float(TOP_Z_MAX),
                            ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION),
                            ctypes.c_float(TOP_Z_DIVISION),
                            ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn)
                            )
    top = np.flipud(np.fliplr(top_flip))
    return top

def load_velo_scans(velo_file):
    # read velo files
    return np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))

def draw_top_image(top, ch3_dir):

    for i in range(top.shape[2]):
        img = top[:, :, i]
        img = img - np.min(img)
        divisor = np.max(img) - np.min(img)
        if divisor!=0 : img = (img / divisor * 255)
        top[:, :, i] = img.astype(np.uint8)

    img_height = top[:, :, :top.shape[2] - 2] 
    img_height = np.sum(img_height, axis=2)
    divisor = np.max(img_height) - np.min(img_height)
    if divisor!=0 : img_height = (img_height / divisor * 255).astype(np.uint8)
#    for i in range(top.shape[2] - 2):
#        img = top[:, :, i]
#        cv2.imwrite('chkheight_%d.png' % i, img)

#    cv2.imwrite('chkheight_sum.png', img_height)
#    cv2.imwrite('chkintensity.png', top[:, :, top.shape[2] - 2])
#    cv2.imwrite('chkdensity.png', top[:, :, top.shape[2] - 1])

    ch3_img = np.dstack((img_height, top[:, :, top.shape[2] - 2], top[:, :, top.shape[2] - 1])).astype(np.uint8)
    cv2.imwrite(ch3_dir, ch3_img)


def lidar_to_top_coords(y,x,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    xx = int((y-TOP_Y_MIN)//TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_DIVISION)

    return xx,yy

def box3d_to_top_box(boxx):

    b = np.zeros((4),  dtype=np.float32)

    boxx = np.array(boxx)
    x0, y0, x1, y1, x2, y2, x3, y3 = boxx[0,0], boxx[0,1], boxx[1,0], boxx[1,1], \
            boxx[2,0], boxx[2,1], boxx[3,0], boxx[3,1]
    u0,v0=lidar_to_top_coords(x0,y0)
    u1,v1=lidar_to_top_coords(x1,y1)
    u2,v2=lidar_to_top_coords(x2,y2)
    u3,v3=lidar_to_top_coords(x3,y3)

    box = np.array([[u0, v0], [u1, v1], [u2, v2], [u3, v3]])

    return box

# main

N = 7481
for i in range(N):
    print("%d/%d" % (i, N))
    gt_dir = "/media/yc/c45eb821-d419-451d-b171-3152a8436ba2/KITTI/data_object_image_2/training/label_2/%06d.txt" % i
    ch3_dir = "./3channel_png/%06d_3ch.png" % i
    chk_dir= "./3channel_gt/%06d_gt.png" % i
    velo_dir = "/media/yc/c45eb821-d419-451d-b171-3152a8436ba2/KITTI/data_object_velodyne/training/velodyne/%06d.bin" % i

    # read velodyne npy file
    lidar = load_velo_scans(velo_dir)
    top = clidar_to_top(lidar)
    draw_top_image(top, ch3_dir)

    # draw bbox in png    
    f = open(gt_dir , 'r')
    cnt = 0
    while True:
        line = f.readline()
        if not line: break
        if line.find('DontCare') != -1: continue
        objects = obj(line.split(' '))
#        objects.showCoor(cnt)
        objects.draw_bbox(ch3_dir, chk_dir)
        cnt += 1
