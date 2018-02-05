# KITTI Velodyne data Top View Converter (Python)
make KITTI velodyne lidar data, label for 3d to top view coordinates

data link: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
reference: https://github.com/bostondiditeam/MV3D, kitti devkit 

## make LidarTopPreprocess.so
`$ g++ -Wall -O3 -shared LidarTopPreprocess.c -o LidarTopPreprocess.so - fPIC`

Compile .c file to use it in python

## run getconer_3D.py
`python getcorner_3D.py`

You need to modify the directory part before you run it

## result

![3ch1](https://github.com/rasd3/KITTI_Topview_Converter/blob/master/image/000006_3ch.png)
![gt1](https://github.com/rasd3/KITTI_Topview_Converter/blob/master/image/000006_gt.png)
![complete](https://github.com/rasd3/KITTI_Topview_Converter/blob/master/image/result.png)
