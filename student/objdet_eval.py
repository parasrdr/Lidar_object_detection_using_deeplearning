# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter
import csv

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            box = label.box
            box_lab = tools.compute_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)
            
            ## step 2 : loop over all detected objects
            for bbox in detections:
                ## step 3 : extract the four corners of the current detection
                bid, x, y, z, h, w, l, yaw = bbox
                box_det = tools.compute_box_corners(x, y, w, l, yaw)
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                dist_x = box.center_x - x
                dist_y = box.center_y - y
                dist_Z = box.center_z - z
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                poly_1 = Polygon(box_lab)
                poly_2 = Polygon(box_det)
                intersection = poly_1.intersection(poly_2).area 
                union = poly_1.union(poly_2).area
                iou = intersection / union
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    matches_lab_det.append([iou,dist_x, dist_y, dist_Z ])
                    true_positives = true_positives + 1
                
                
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = labels_valid.sum()

    ## step 2 : compute the number of false negatives
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives
    false_positives =  len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all,configs):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
        pos_negs_arr = np.asarray(pos_negs)
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    positives = sum(pos_negs_arr[:,0])
    true_positives = sum(pos_negs_arr[:,1])
    false_negatives = sum(pos_negs_arr[:,2])
    false_positives = sum(pos_negs_arr[:,3])
    
    ## step 2 : compute precision
    precision = true_positives /float(true_positives + false_positives)  
    precision_rounded = round(precision,2)
    ## step 3 : compute recall 
    recall = true_positives / float(true_positives + false_negatives)
    recall_rounded = round(recall,2)


    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))  
    
    result ={'precision': precision_rounded,'recall': recall_rounded,'confidence_threshold': configs.conf_thresh,'IOU':configs.min_iou}
    
    with open('precision_recall.csv','a',newline='') as file_object:
       
        fieldnames = ['precision','recall','confidence_threshold','IOU']
        csv_dict_writer = csv.DictWriter(file_object,delimiter =',',fieldnames=fieldnames)
        
        csv_dict_writer.writerow(result)
    
    
    
   

