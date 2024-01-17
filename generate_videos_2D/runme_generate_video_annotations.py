import os
import torch
import pdb
import pycocotools.mask as cocomask
from tqdm import tqdm
import numpy as np
import json
import pickle

data = {}
data['info'] = ''
data['description'] = ''
data['categories'] = []
data['categories'].append({'supercategory': 'object', 'id': 1, 'name': 'fish'})
data['videos'] = []
data['annotations'] = []
current_path = ''

####
width = 640
height = 640
vid_files = os.listdir('danio2d_small/val/images/')
num_videos = len(vid_files)
####

annotation_idx = 1
for vid_idx in tqdm(range(num_videos)):
#for vid_idx in range(2):
    filenames = []
    vid_rootpath = os.path.join(current_path, 'danio2d_small/val')
    frames = os.listdir(os.path.join(vid_rootpath, 'images', vid_files[vid_idx]))
    num_frames = len(frames)
    for frame_idx in range(len(frames)):
        filenames.append(os.path.join(vid_files[vid_idx], frames[frame_idx]))
    filenames = sorted(filenames)
    videos_key = {'license':'', 'coco_url': '', 'height': height, 'width': width, 'length': num_frames,\
            'date_captured': '0000-00-00 00:00:00.903902', 'file_names': filenames, 'flickr_url': '', 'id': vid_idx + 1}
    data['videos'].append(videos_key)
    num_annotations = 1
    
    bboxes = []
    bboxes_path = os.path.join(vid_rootpath, 'annotations', vid_files[vid_idx], 'bbox')
    bboxes_files_list = sorted(os.listdir(bboxes_path))
    segmentations = []
    segmentations_path = os.path.join(vid_rootpath, 'annotations', vid_files[vid_idx], 'segmentations')
    segmentations_files_list = sorted(os.listdir(segmentations_path))
    
    areas = []
    for category_idx in range(num_annotations):
        for bbox_idx in range(len(bboxes_files_list)):
            #bbox = torch.load(os.path.join(bboxes_path, bboxes_files_list[bbox_idx]))
            #bboxes.append(bbox.tolist())
            with open(os.path.join(segmentations_path, segmentations_files_list[bbox_idx]), "rb") as fp:
                segmentation_annotation = pickle.load(fp)[category_idx]
            segmentation_dict = {'counts': segmentation_annotation['segmentations']['counts'].decode('ascii'), 'size': segmentation_annotation['segmentations']['size']}
            segmentations.append(segmentation_dict)
            (segmentation_annotation["segmentations"])
            bbox=(cocomask.toBbox(segmentation_annotation["segmentations"]))
            bboxes.append(bbox.tolist())
            areas.append(int(segmentation_annotation['area']))
        annotations_dict = {'video_id': vid_idx + 1, 'iscrowd': 0, 'height': height, 'width': width, \
                    'length': 1, 'bboxes': bboxes, 'category_id': category_idx + 1, 'id': annotation_idx, 'segmentations': segmentations, 'areas': areas}
        data['annotations'].append(annotations_dict)
        annotation_idx += 1 


with open('val.json', 'w') as json_file:
    pdb.set_trace()
    json.dump(data, json_file)
