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
import pickle
import torch
import argparse
from tqdm import tqdm
import os
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pycocotools import mask as maskUtils
from itertools import groupby
import pdb
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

def binary_image_to_coco_annotations(binary_image):
    # Find contours in the binary image
    # binary image shape: [h, w, n], where n is the number of fishes
    if len(binary_image.shape)==2:
        binary_image = np.expand_dims(binary_image, axis=-1) 
    segmentations = maskUtils.encode(np.asfortranarray(binary_image))
    areas = maskUtils.area(segmentations)
    #contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #segmentations = []
    #areas = []
    #pdb.set_trace()
    #for contour in contours:
    #    areas.append(int(cv2.contourArea(contour)))
    #    # Flatten the contour coordinates into a 1D array
    #    contour = contour.flatten().tolist()
        # Convert the 1D array to a list of (x, y) pairs
    #    contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
        # Convert the (x, y) pairs to a list of integers
    #    segmentations.append([int(coord) for pair in contour_pairs for coord in pair])
    return segmentations, areas

############################################################
random.seed(10)
render_spots = False # Render spots that look like paramecia
randomize = False # Randomize different parts of the fish
averageFishLength = 67
imageSizeX = 640
imageSizeY = 640
bufferX = 30
bufferY = 30
############################################################
for mat_file_idx in tqdm(range(0, 1)):
    # Load the array of pose angles : generated_pose_all_2D_50k
    # The angles are sampled from a probability distribution learnt from the  distribution of real poses (see manuscript)
    filepath = os.path.join(trajectory_folder, str(mat_file_idx).rjust(3, '0') + '.mat')
    x_vid = sio.loadmat(filepath)
    x_vid = x_vid['x_vid']
    data_folder = os.path.join(data_folder_parent, str(mat_file_idx).rjust(3, '0'))
    image_folder = os.path.join(data_folder_parent, 'images', str(mat_file_idx).rjust(3, '0'))
    annotations_folder = os.path.join(data_folder_parent, 'annotations', str(mat_file_idx).rjust(3, '0'))
    #print(mat_file_idx)
    if not os.path.exists(data_folder_parent) or not os.path.exists(image_folder) or not os.path.exists(annotations_folder):
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(annotations_folder + '/coor_2d')
        os.makedirs(annotations_folder + '/bbox')
        os.makedirs(annotations_folder + '/segmentations')
    else:
        print('Warning: data_folder already exists. Files might be overwritten')

    n_frames = x_vid.shape[0]
    x_vid[:, 0] = x_vid[:, 0].copy() + imageSizeX / 2 
    x_vid[:, 1] = x_vid[:, 1].copy() + imageSizeY / 2
    x_vid[:, 2] = x_vid[:, 2].copy() + np.random.rand() * 2 * np.pi
    theta_0 = x_vid[0, 2] .copy()
    rot_mat = R.from_euler('z', theta_0, degrees=False).as_matrix()
    xy_traces = np.dot(rot_mat[0:2, 0:2], np.append([x_vid[:,0]], [x_vid[:,1]], axis=0))
    x_vid[:, 0] = xy_traces[0, :].copy()
    x_vid[:, 1] = xy_traces[1, :].copy()
    shift_x = [np.min(x_vid[:, 0]) - bufferX, imageSizeX - np.max(x_vid[:, 0]) - bufferX]
    shift_y = [np.min(x_vid[:, 1]) - bufferY, imageSizeY - np.max(x_vid[:, 1]) - bufferY]
    
    x_vid[:, 0] = x_vid[:, 0].copy() + np.random.randint(-shift_x[0], shift_x[1])
    x_vid[:, 1] = x_vid[:, 1].copy() + np.random.randint(-shift_y[0], shift_y[1])

    fishlen = (np.random.rand(1) - 0.5) * 15 + averageFishLength
    #for frame in tqdm(range(0, n_frames)):
    video_file_path = os.path.join(data_folder_parent, 'vid_' + str(mat_file_idx).rjust(3, '0') + '.mp4')
    writer = imageio.get_writer(video_file_path, fps = 60)
    for frame in range(0, n_frames):
        if ((x_vid[frame, 0] > imageSizeX - bufferX) or (x_vid[frame, 1] > imageSizeY - bufferY) or (x_vid[frame, 0] < bufferX) or (x_vid[frame, 1] < bufferY)):
            continue
        # Initate random orientation of the larva
        x = x_vid[frame, :].copy()
        #x = x_vid[frame, np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
        # Sample a fish length to generate physical model image
        idxlen = np.floor((fishlen - 62) / 1.05) + 1
        seglen = 5.6 + idxlen * 0.1
        # Render physical model image and the corresponding annotation
        graymodel, pt = f_x_to_model(x.T, seglen, randomize)
        y, x = np.argwhere(graymodel > 0)[0]
        y = np.maximum(0, y - np.random.randint(0, 5))
        x = np.maximum(0, x - np.random.randint(0, 5))
        y_max, x_max = np.argwhere(graymodel > 0)[-1]
        y_max = np.minimum(graymodel.shape[1], y_max + np.random.randint(0, 5))
        x_max = np.minimum(graymodel.shape[0], x_max + np.random.randint(0, 5))
        width = x_max - x
        height = y_max - y
        bbox = torch.tensor([x, y, width, height])
        graymodel = np.uint8(255 * (graymodel / np.max(graymodel)))

        _, thresholded_image = cv2.threshold(graymodel, 0, 255, cv2.THRESH_BINARY)
        segmentations, areas = binary_image_to_coco_annotations(thresholded_image)
        
        segmentation_annotation = []
        for segmentation, area in zip(segmentations, areas):
            segmentation_annotation.append({'segmentations': segmentation, 'area': area})
        if render_spots:
            for ellipse_idx in range(0, np.random.randint(4)):
                ellipse_x = int(graymodel.shape[1]*np.random.rand(1))
                ellipse_y = int(graymodel.shape[0]*np.random.rand(1))
                ellipse_axis_x = 1 + int(3*np.random.rand(1))
                ellipse_axis_y = 1 + int(3*np.random.rand(1))
                ellipse_angle = int(np.random.rand(1)*360)
                ellipse_two_angles = [0, 360]
                ellipse_color = int(120 * np.random.rand(1))
                graymodel = cv2.ellipse(graymodel,(ellipse_x, ellipse_y), (ellipse_axis_x, ellipse_axis_y), ellipse_angle, 
                        ellipse_two_angles[0], ellipse_two_angles[1], (ellipse_color, ellipse_color, ellipse_color), -1)
        if add_Gaussian_noise:
            graymodel = add_noise('gauss', graymodel, 0.001 * 100 * random.rand(1), 0.002 * 5 * random.rand(1))
        graymodel = np.uint8(graymodel)
        pt = torch.tensor(pt, dtype=torch.float32)
        
        
        # Append frame to video file
        writer.append_data(np.uint8(graymodel))


        # Save rendered image and annotation
        cv2.imwrite(image_folder + '/im_' + str(frame).rjust(6, "0") + '.png', np.uint8(graymodel))
        torch.save(bbox, annotations_folder + '/bbox/bbox_' + str(frame).rjust(6, "0") + '.pt')
        with open(annotations_folder + '/segmentations/segmentation_' + str(frame).rjust(6, "0"), "wb") as fp:   #Pickling
            pickle.dump(segmentation_annotation, fp)
        torch.save(pt, annotations_folder + '/coor_2d/ann_' + str(frame).rjust(6, "0") + '.pt')
    writer.close()

print('Finished generating data')

