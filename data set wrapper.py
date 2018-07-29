# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:39:03 2018

@author: dell
"""
import numpy as np
import cv2
import zlib
import math
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

BATCH_SIZE=32
HEIGHT=192
WIDTH=256
POINTCLOUDSIZE=16384
OUTPUTPOINTS=1024
REEBSIZE=1024

path='C:/New folder/python2.7/PointSetGeneration-master/data/0/0.gz'

root_dir='C:/New folder/python2.7/PointSetGeneration-master/data/'
class shapeneteDataset(Dataset):
    
    def __init__(self, path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the gz file with annotations.
            root_dir (string): Directory with all the zip folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.binfile = zlib.decompress(open(path,'rb').read())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.self.binfile)

    def __getitem__(self, idx):
 #       img_name = os.path.join(self.root_dir,
  #                              self.landmarks_frame.iloc[idx, 0])
        p=0
        color=np.fromstring(self.binfile[p:p+BATCH_SIZE*HEIGHT*WIDTH*3],dtype='uint8').reshape((BATCH_SIZE,HEIGHT,WIDTH,3))
        p+=BATCH_SIZE*HEIGHT*WIDTH*3
        depth=np.fromstring(self.binfile[p:p+BATCH_SIZE*HEIGHT*WIDTH*2],dtype='uint16').reshape((BATCH_SIZE,HEIGHT,WIDTH))
        p+=BATCH_SIZE*HEIGHT*WIDTH*2
        rotmat=np.fromstring(self.binfile[p:p+BATCH_SIZE*3*3*4],dtype='float32').reshape((BATCH_SIZE,3,3))
        p+=BATCH_SIZE*3*3*4
        ptcloud=np.fromstring(self.binfile[p:p+BATCH_SIZE*POINTCLOUDSIZE*3],dtype='uint8').reshape((BATCH_SIZE,POINTCLOUDSIZE,3))
        ptcloud=ptcloud.astype('float32')/255
        beta=math.pi/180*20
        viewmat=np.array([[
    		np.cos(beta),0,-np.sin(beta)],[
    		0,1,0],[
    		np.sin(beta),0,np.cos(beta)]],dtype='float32')
        rotmat=rotmat.dot(np.linalg.inv(viewmat))
        for i in range(BATCH_SIZE):
            ptcloud[i]=((ptcloud[i]-[0.7,0.5,0.5])/0.4).dot(rotmat[i])+[1,0,0]
        p+=BATCH_SIZE*POINTCLOUDSIZE*3
        some_other_thing=np.fromstring(self.binfile[p:p+BATCH_SIZE*REEBSIZE*2*4],dtype='uint16').reshape((BATCH_SIZE,REEBSIZE,4))
        p+=BATCH_SIZE*REEBSIZE*2*4
        keynames=self.binfile[p:].split('\n')
        data=np.zeros((BATCH_SIZE,HEIGHT,WIDTH,4),dtype='float32')
        data[:,:,:,:3]=color*(1/255.0)
        data[:,:,:,3]=depth==0
        validating=np.array([i[0]=='f' for i in keynames],dtype='float32')
        sample = {'color': color, 'depth': depth, 'point cloud': ptcloud}

        if self.transform:
            sample = self.transform(sample)

        return sample