# This script generates training dataset for pose estimation

import scipy.io as sio
import numpy as np
#from avi_r import AVIReader
#from preprocessing import bgsub
#from preprocessing import readVideo
from construct_model import f_x_to_model, add_noise
import matplotlib.pyplot as plt
#import time
import numpy.random as random
import os
import scipy.io as sio
import pdb
import cv2
import torch
import argparse
from tqdm import tqdm
import os
import imageio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_folder', default="training_dataset", type=str, help='path to store training dataset')
parser.add_argument('-n','--n_frames',default=500000, type=int, help='number of training examples')
parser.add_argument('-g', '--add_Gaussian_noise', action='store_true', help='add Gaussian background noise to the synthetic image')
parser.add_argument('-t', '--trajectory_folder', default='x_files_5', type=str, help='path to the folder containing trajectories mat files')

# User inputs
args = vars(parser.parse_args())
data_folder_parent = args['data_folder']
n_frames = args['n_frames']
add_Gaussian_noise = args['add_Gaussian_noise']
trajectory_folder = args['trajectory_folder']

############################################################
random.seed(10)
render_spots = False # Render spots that look like paramecia
randomize = False # Randomize different parts of the fish
averageFishLength = 73
############################################################
for mat_file_idx in tqdm(range(0, 600)):
    # Load the array of pose angles : generated_pose_all_2D_50k
    # The angles are sampled from a probability distribution learnt from the  distribution of real poses (see manuscript)
    filepath = os.path.join(trajectory_folder, str(mat_file_idx).rjust(3, '0') + '.mat')
    x_vid = sio.loadmat(filepath)
    x_vid = x_vid['x_vid']
    data_folder = os.path.join(data_folder_parent, str(mat_file_idx).rjust(3, '0'))
    if not os.path.exists(data_folder) or not os.path.exists(data_folder + '/images'):
        os.makedirs(data_folder, exist_ok = True)
        os.makedirs(data_folder + '/images')
        os.makedirs(data_folder + '/coor_2d')
    else:
        print('Warning: data_folder already exists. Files might be overwritten')

    n_frames = x_vid.shape[0]
    x_vid[:, 0] = x_vid[:, 0].copy() + 320
    x_vid[:, 1] = x_vid[:, 1].copy() + 320
    fishlen = (np.random.rand(1) - 0.5) * 15 + averageFishLength
    #for frame in tqdm(range(0, n_frames)):
    video_file_path = os.path.join(data_folder_parent, 'vid_' + str(mat_file_idx).rjust(3, '0') + '.mp4')
    writer = imageio.get_writer(video_file_path, fps = 60)
    for frame in range(0, n_frames):
        # Initate random orientation of the larva
        x = x_vid[frame, :].copy()
        #x = x_vid[frame, np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
        # Sample a fish length to generate physical model image
        idxlen = np.floor((fishlen - 62) / 1.05) + 1
        seglen = 5.6 + idxlen * 0.1
        # Render physical model image and the corresponding annotation
        graymodel, pt = f_x_to_model(x.T, seglen, randomize)
        graymodel = np.uint8(255 * (graymodel / np.max(graymodel)))
        if render_spots:
            #for i in range(0,4):
            for ellipse_idx in range(0, np.random.randint(4)):
                ellipse_x = int(graymodel.shape[1]*np.random.rand(1))
                ellipse_y = int(graymodel.shape[0]*np.random.rand(1))
                ellipse_axis_x = 1 + int(3*np.random.rand(1))
                ellipse_axis_y = 1 + int(3*np.random.rand(1))
                ellipse_angle = int(np.random.rand(1)*360)
                ellipse_two_angles = [0, 360]
                ellipse_color = int(120*np.random.rand(1))
                graymodel = cv2.ellipse(graymodel,(ellipse_x, ellipse_y), (ellipse_axis_x, ellipse_axis_y), ellipse_angle, 
                        ellipse_two_angles[0], ellipse_two_angles[1], (ellipse_color, ellipse_color, ellipse_color), -1)
        if add_Gaussian_noise:
            graymodel = add_noise('gauss', graymodel, 0.001 * 100 * random.rand(1), 0.002 * 5 * random.rand(1))
        graymodel = np.uint8(graymodel)
        pt = tensor = torch.tensor(pt, dtype=torch.float32)
        
        
        # Append frame to video file
        writer.append_data(np.uint8(graymodel))


        # Save rendered image and annotation
        #cv2.imwrite(data_folder + '/images/im_' + str(frame).rjust(6, "0") + '.png', np.uint8(graymodel))
        #torch.save(pt, data_folder + '/coor_2d/ann_' + str(frame).rjust(6, "0") + '.pt')
    writer.close()

print('Finished generating data')

