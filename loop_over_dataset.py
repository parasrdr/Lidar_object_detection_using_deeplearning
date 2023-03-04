# ---------------------------------------------------------------------
# Project "3D object Detection Lidar"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Loop over all frames in a Waymo Open Dataset file,
#                        detect objects and visualize results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
# Modified by: Paras Arora

##################
## Imports

## general package imports
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

## 3d object detection
import student.objdet_pcl as pcl
import student.objdet_detect as det
import student.objdet_eval as eval

import misc.objdet_tools as tools 
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list


## Waymo Open Dataset Test data sequence.Change data_filename if you want to Test on different test sequences
#data_filename = 'testing_segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord' # Test Sequence 1

#Training and validation sequences used for training fsn_resnet_waymo_model

#Training data

#data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
data_filename = 'training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord' #Sequene 2
#data_filename = 'training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord' #Sequence 3

# Validation Data
#data_filename = 'validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord'

show_only_frames = [0, 200] # show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator

## Initialize object detection
configs_det = det.load_configs(model_name='fpn_resnet') # options are 'darknet', 'fpn_resnet','fpn_resnet_waymo'
model_det = det.create_model(configs_det)

configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

## Selective execution and visualization
exec_detection = ['pcl_from_rangeimage','bev_from_pcl','detect_objects','validate_object_labels', 'measure_detection_performance'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_tracking =[]
exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_detection_performance'
exec_list = make_exec_list(exec_detection,exec_tracking,exec_visualization)
vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed)


##################
## Perform detection over all selected frames

cnt_frame = 0 
all_labels = []
det_performance_all = [] 
np.random.seed(0) # make random values predictable
if 'show_tracks' in exec_list:    
    fig, (ax2, ax) = plt.subplots(1,2) # init track plot

while True:
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        if cnt_frame < show_only_frames[0]:
            cnt_frame = cnt_frame + 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break
        
        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        #################################
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)        
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) 

        ## Compute lidar point-cloud from range image    
        if 'pcl_from_rangeimage' in exec_list:
            print('computing point-cloud from lidar range image')
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name)
        else:
            print('loading lidar point-cloud from result file')
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame)
        
        print(lidar_pcl.shape)    
        ## Compute lidar birds-eye view (bev)
        if 'bev_from_pcl' in exec_list:
            print('computing birds-eye view from lidar pointcloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)
            
        else:
            print('loading birds-eve view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', cnt_frame)

        ## 3D object detection/Inference mode for our Deep learning models.
        
        if (configs_det.use_labels_as_objects==True):
            print('using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list:
                print('detecting objects in lidar pointcloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det)
            else:
                print('loading detected objects from result file')
                # load different data for final project vs. mid-term project
                detections = load_object_from_file(results_fullpath, data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)

        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("validating object labels")
           
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10)
           
        else:
            print('loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', cnt_frame)            

        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('measuring detection performance')
            det_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou)   
            
        else:
            print('loading detection performance measures from file')
            # load different data for final project vs. mid-term project
            if 'perform_tracking' in exec_list:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance', cnt_frame)
            else:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)   

        det_performance_all.append(det_performance) # store all evaluation results in a list for performance assessment at the end
        

        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            img_range = pcl.show_range_image(frame, lidar_name)
            img_range = img_range.astype(np.uint8)
            cv2.imshow('range_image', img_range)
            cv2.waitKey(vis_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_pcl)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)  
            cv2.waitKey(vis_pause_time)          

        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(vis_pause_time)

        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(detections, frame.laser_labels, lidar_bev, configs_det)
            cv2.waitKey(vis_pause_time)         

        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(detections, lidar_bev, image, frame.laser_labels, valid_label_flags, camera_calibration, configs_det)
            cv2.waitKey(vis_pause_time)               


    
        # increment frame counter
        cnt_frame = cnt_frame + 1    

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break


#################################
## Post-processing

## Evaluate object detection performance

eval.compute_performance_stats(det_performance_all, configs_det)

 