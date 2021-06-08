'''
Inference code for PropTR
Modified from VisTR (https://github.com/Epiphqny/VisTR)
'''
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_util

from torchvision.utils import save_image
import pdb


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--img_path', default='/n/pfister_lab2/Lab/vcg_natural/YouTube-VIS/valid/JPEGImages/')
    parser.add_argument('--ann_path', default='/n/pfister_lab2/Lab/vcg_natural/YouTube-VIS/vis/valid.json')
    parser.add_argument('--save_path', default='results.json')
    parser.add_argument('--dataset_file', default='ytvos')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_ytvos',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

CLASSES = ['person','giant_panda','lizard','parrot','skateboard','sedan','ape',
           'dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
           'train','horse','turtle','bear','motorbike','giraffe','leopard',
           'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
           'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
           'tennis_racket']

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600]]

transform = T.Compose([
    T.Resize(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def sequence_nms(key_masks, key_scores, key_labels):
    K, num_queries = key_scores.shape
    length = key_masks.shape[1]
    score_thr = 0.001
    iou_thr = 0.5

    scores = key_scores.flatten()
    areas = np.sum(key_masks, axis=(-1, -2, -3))
    order = scores.argsort()[::-1]
    
    keep, labels = [], []
    while order.size > 0:
        i = order[0]
        score = scores[i]
        if score <= score_thr:
            break
        keep.append(i)

        inter = np.bitwise_and(key_masks[i], key_masks[order[1:]]).sum(axis=(-1,-2,-3))
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # majority voting for sequence category
        pos_inds = np.where(ovr > iou_thr)[0]
        pos_labels = np.append(key_labels[order[pos_inds]], key_labels[i])
        label = np.bincount(pos_labels).argmax().item()
        if label <= 0:
            pdb.set_trace()
        labels.append(label)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep, labels


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    num_queries = args.num_queries
    K = 4
    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)
        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)
        model.eval()
        folder = args.img_path
        videos = json.load(open(args.ann_path, 'rb'))['videos']
        vis_num = len(videos)
        result = []
        for i in range(vis_num):
            print("Process video: ", i)
            id_ = videos[i]['id']
            length = videos[i]['length']
            file_names = videos[i]['file_names']
            # propose reduce results
            key_frames = [max(length//K, 1) * k for k in range(K)]
            key_scores = np.zeros((K, num_queries, length), dtype=float)
            key_labels = np.zeros((K, num_queries, length), dtype=np.uint8)
            key_masks = None
            for k_id, frame_id in enumerate(key_frames):
                ref_im = Image.open(os.path.join(folder, file_names[frame_id])).convert('RGB')
                ref_img = transform(ref_im).unsqueeze(0).cuda()
                if key_masks is None:
                    key_masks = np.zeros((K, num_queries, length, *ref_im.size[1::-1]), dtype=np.uint8)
                prev_img = ref_img
                for t in range(length):
                    if t == frame_id == 0:
                        outputs = model(ref_img)
                    else:
                        im = Image.open(os.path.join(folder, file_names[t])).convert('RGB')
                        img = transform(im).unsqueeze(0).cuda()
                        outputs = model(img, prev_img)
                        prev_img = img
                    logits, boxes, masks = (outputs['pred_logits'].softmax(-1)[0,:,:-1], 
                                            outputs['pred_boxes'][0], outputs['pred_masks'][0])
                    pred_masks = F.interpolate(
                        masks.unsqueeze(0), size=ref_im.size[1::-1], mode="bilinear")[0]
                    pred_masks = pred_masks.sigmoid().cpu().detach().numpy() > 0.5
                    pred_logits = logits.cpu().detach().numpy()
                    pred_scores = np.max(pred_logits,axis=-1)
                    pred_logits = np.argmax(pred_logits,axis=-1)
                    key_scores[k_id, :, t] = pred_scores
                    key_labels[k_id, :, t] = pred_logits
                    key_masks[k_id, :, t] = pred_masks.astype(np.uint8)
            for i_id in range(num_queries):
                scores, segmentation = [], []
                label = np.bincount(key_labels[:, i_id, :].flatten()).argmax()
                for t in range(length):
                    idx = key_scores[:, i_id, t].argmax()
                    score = key_scores[idx, i_id, t]
                    # scores.append(score)
                    mask = key_masks[idx, i_id, t]
                    if mask.max() == 0:
                        segmentation.append(None)
                    else:
                        rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
                        rle["counts"] = rle["counts"].decode("utf-8")
                        segmentation.append(rle)
                        scores.append(score)
                if segmentation.count(None) == length:
                    continue
                score = np.mean(scores)
                instance = {'video_id': id_, 'score': float(score), 'category_id': int(label)}
                instance['segmentations'] = segmentation
                result.append(instance)
            # key_scores = key_scores.mean(axis=2)
            # key_masks = key_masks.reshape(-1, length, *key_masks.shape[-2:])
            # key_labels = key_labels.reshape(-1, length)
            # key_labels = np.asarray([
            #     np.argmax(np.bincount(key_labels[i]))
            #     for i in range(key_labels.shape[0])
            # ])
            # keep, labels = sequence_nms(key_masks, key_scores, key_labels)
            # scores = key_scores.flatten()
            # for obj_id, idx in enumerate(keep):
            #     label = labels[obj_id]
            #     score = scores[obj_id]
            #     segmentation = []
            #     instance = {'video_id': id_, 'score': float(score), 'category_id': int(label)}
            #     for t in range(length):
            #         mask = key_masks[obj_id, t]
            #         if mask.max() == 0:
            #             segmentation.append(None)
            #         rle = mask_util.encode(np.array(mask[:,:,np.newaxis], order='F'))[0]
            #         rle["counts"] = rle["counts"].decode("utf-8")
            #         segmentation.append(rle)
            #     instance['segmentations'] = segmentation
            #     result.append(instance)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
