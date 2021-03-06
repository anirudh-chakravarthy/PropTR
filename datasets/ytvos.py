"""
YoutubeVIS data loader

Adapted from https://github.com/Epiphqny/VisTR/blob/3f736292330424f53905bdcfb1cdf07cc2902eb5/datasets/ytvos.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random

class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid, frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        # load image
        img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id])
        img = Image.open(img_path).convert('RGB')
        # * reference image
        _, ref_frame_id = self.sample_ref(idx)
        ref_img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][ref_frame_id])
        ref_img = Image.open(ref_img_path).convert('RGB')
        # load annotations
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img, target)
        # * reference frame annotations
        ref_idx = self.img_ids.index((vid, ref_frame_id))
        ref_target = self.ytvos.loadAnns(ann_ids)
        ref_target = {'image_id': ref_idx, 'video_id': vid, 'frame_id': ref_frame_id, 'annotations': ref_target}
        ref_target = self.prepare(ref_img, ref_target)
        if self._transforms is not None:
            # TODO: need to pass both together
            img, ref_img, target, ref_target = self._transforms(img, ref_img, target, ref_target)
        return torch.tensor(img), torch.tensor(ref_img), target, ref_target
    
    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = self.img_ids[idx]
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
#         if frame_id > 0:
#             ref_idx = (vid, frame_id-1)
#         else:
#             assert frame_id+1 < len(vid_info['filenames'])
#             ref_idx = (vid, frame_id+1)
#         return ref_idx
        valid_samples = []
        for i in sample_range:
            # check if the frame id is valid
            ref_idx = (vid, i)
            if i != frame_id and ref_idx in self.img_ids:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            bbox = ann['bboxes'][frame_id]
            areas = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            clas = ann["category_id"]
            # for empty boxes
            if bbox is None:
                bbox = [0,0,0,0]
                areas = 0
                valid.append(0)
                clas = 0
            else:
                valid.append(1)
            crowd = ann["iscrowd"] if "iscrowd" in ann else 0
            boxes.append(bbox)
            area.append(areas)
            segmentations.append(segm)
            classes.append(clas)
            iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     T.RandomResize([360], max_size=640),
                     # T.RandomResize([320], max_size=576),
                     # To suit the GPU memory the scale might be different
                     # T.RandomResize([300], max_size=540),#for r50
                     # T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/JPEGImages", root / "vis" / 'train.json'),
        "val": (root / "valid/JPEGImages", root / "vis" / 'valid.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset