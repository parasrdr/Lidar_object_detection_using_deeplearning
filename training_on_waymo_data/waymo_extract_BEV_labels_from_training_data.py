"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset
"""

import sys
import os
import math
from builtins import int
from pathlib import Path  
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
    

import misc.objdet_tools as tools 

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2


from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
from data_process import transformation
import config.kitti_config as cnf



## Select Waymo Open Dataset file and frame numbers

#Training data

data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
#data_filename = 'training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord' #Sequene 2
#data_filename = 'training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord' #Sequence 3



# Validation Data
#data_filename = 'validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'

validation =False
if(data_filename == 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'):
    waymo_training_sequence = 1

elif(data_filename == 'training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord'):
    waymo_training_sequence = 2

elif(data_filename == 'validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'):
    waymo_training_sequence = 1
    validation = True
else:
    waymo_training_sequence = 3
    
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator

cnt_frame = 0
configs = edict()    
configs.lim_x = [0, 50] # detection range in m
configs.lim_y = [-25, 25]
configs.lim_z = [-1, 3]
configs.lim_r = [0, 1.0] # reflected lidar intensity
configs.bev_width = 608  # pixel resolution of bev image
configs.bev_height = 608 

TRAINING_DATA_PATH =Path("SFA3D/dataset/Waymo/Training/Lidar_BEV")
TRAINING_DATA_PATH.mkdir(parents=True,exist_ok=True)
TRAINING_LABELS_PATH =Path("SFA3D/dataset/Waymo/Training/Lidar_3D_Labels")
TRAINING_LABELS_PATH.mkdir(parents=True,exist_ok=True)

VALIDATION_DATA_PATH =Path("SFA3D/dataset/Waymo/Validation/Lidar_BEV")
VALIDATION_DATA_PATH.mkdir(parents=True,exist_ok=True)
VALIDATION_LABELS_PATH =Path("SFA3D/dataset/Waymo/Validation/Lidar_3D_Labels")
VALIDATION_LABELS_PATH.mkdir(parents=True,exist_ok=True)



while True:
    try:
        ## Get next frame from Waymo dataset (iterate for each frame)
        frame = next(datafile_iter)
        
        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
      
        labels_currFrame = tools.convert_labels_into_objects(frame.laser_labels,configs) 
        ## Compute lidar point-cloud from range image    
       
        print('computing point-cloud from lidar range image')
            
        lidarData = tools.pcl_from_range_image(frame, lidar_name)
        
        lidarData = get_filtered_lidar(lidarData, cnf.boundary)
        print('Converting into BEV MAP')
        bev_map = makeBEVMap(lidarData, cnf.boundary)
     
        
        bev_string = f'bev_map_' + f'sequence{waymo_training_sequence}_frame{cnt_frame}'
        label_string = f'label_'+ f'sequence{waymo_training_sequence}_frame{cnt_frame}'
        
        print('Storing BEV Maps & Labels in dataset/Waymo Directory')
        if validation:
            BEV_SAVE_PATH = VALIDATION_DATA_PATH / Path(bev_string)
            LABEL_SAVE_PATH = VALIDATION_LABELS_PATH / Path(label_string)
            np.save(BEV_SAVE_PATH,bev_map)
            np.save(LABEL_SAVE_PATH,labels_currFrame)  
        
        else:
            BEV_SAVE_PATH = TRAINING_DATA_PATH / Path(bev_string)
            LABEL_SAVE_PATH = TRAINING_LABELS_PATH / Path(label_string)
            np.save(BEV_SAVE_PATH,bev_map)
            np.save(LABEL_SAVE_PATH,labels_currFrame)

        cnt_frame = cnt_frame + 1    

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break
     
            
        

