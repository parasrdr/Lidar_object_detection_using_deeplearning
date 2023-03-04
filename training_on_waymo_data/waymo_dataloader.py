"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.waymo_dataset import WaymoDataset
from data_process.transformation import OneOf, Random_Rotation, Random_Scaling


def create_train_dataloader(configs):
    """Create dataloader for training"""
    
    train_dataset = WaymoDataset(configs, mode='train')
                                 
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                                  pin_memory=configs.pin_memory)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = WaymoDataset(configs, mode='val')
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

    return val_dataloader

