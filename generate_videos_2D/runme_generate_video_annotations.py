import os
import torch
import pdb
from tqdm import tqdm
import json

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
vid_files = os.listdir('danio2d_small/images/')
num_videos = len(vid_files)
####

for vid_idx in tqdm(range(num_videos)):
#for vid_idx in range(2):
    filenames = []
    vid_rootpath = os.path.join(current_path, 'danio2d_small/')
    frames = os.listdir(os.path.join(vid_rootpath, 'images', vid_files[vid_idx]))
    num_frames = len(frames)
    for frame_idx in range(len(frames)):
        filenames.append(os.path.join(vid_rootpath, 'images', vid_files[vid_idx], frames[frame_idx]))

    videos_key = {'license':'', 'coco_url': '', 'height': height, 'width': width, 'length': num_frames,\
            'date_captured': '0000-00-00 00:00:00.903902', 'file_names': filenames, 'flickr_url': '', 'id': vid_idx + 1}
    data['videos'].append(videos_key)
    num_annotations = 1
    
    bboxes = []
    bboxes_path = os.path.join(vid_rootpath, 'annotations', vid_files[vid_idx], 'bbox')
    bboxes_files_list = os.listdir(bboxes_path)
    for annotation_idx in range(num_annotations):
        for bbox_idx in range(len(bboxes_files_list)):
            bbox = torch.load(os.path.join(bboxes_path, bboxes_files_list[bbox_idx]))
            bboxes.append(bbox.tolist())
        annotations_dict = {'video_id': vid_idx + 1, 'iscrowd': 0, 'height': height, 'width': width,\
                'length': 1, 'bboxes': bboxes, 'category_id': 1, 'id': annotation_idx + 1, 'segmentation': '', 'areas': []}
        data['annotations'].append(annotations_dict)

with open('annotations.json', 'w') as json_file:
    json.dump(data, json_file)
