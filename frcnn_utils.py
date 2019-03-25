import sys

sys.path.insert(0, "./frcnn")

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import numpy as np
from model.roi_layers import nms

from torch.utils.data.sampler import Sampler
import torch
import os
import time

from utils import ParameterType
from prune_utils import get_prune_index_target_with_reset


class FasterRCNN_prepare():
    def __init__(self, net, batch_size_train):
        self.lr_decay_step = 5
        self.lr_decay_gamma = 0.1
        self.max_per_image = 100
        self.thresh = 0.0 #0.0 for computing score, change to higher for visualization
        self.class_agnostic = False
        self.save_dir = "myTmp"
        self.large_scale = False
        self.dataset = "pascal_voc"
        self.net = net
        self.batch_size_train = batch_size_train
        self.batch_size_test = 1

    def forward(self):
        if self.dataset == "pascal_voc":
            imdb_name = "voc_2007_trainval"
            imdbval_name = "voc_2007_test"
            set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif self.dataset == "pascal_voc_0712":
            imdb_name = "voc_2007_trainval+voc_2012_trainval"
            imdbval_name = "voc_2007_test"
            set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        cfg_file = "frcnn/cfgs/{}_ls.yml".format(self.net) if self.large_scale else "frcnn/cfgs/{}.yml".format(self.net)
        if cfg_file is not None:
            cfg_from_file(cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)

        cfg.TRAIN.USE_FLIPPED = True
        cfg.CUDA = True
        cfg.USE_GPU_NMS = True

        self.imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(imdb_name)
        self.train_size = len(roidb_train)
        self.imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb(imdbval_name, False)
        self.imdb_test.competition_mode(on=True)

        output_dir = self.save_dir + "/" + self.net + "/" + self.dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sampler_batch = sampler(self.train_size, self.batch_size_train)
        dataset_train = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, self.batch_size_train, \
                                       self.imdb_train.num_classes, training=True)
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size_train,
                                                       sampler=sampler_batch, num_workers=0)
        save_name = 'faster_rcnn_{}'.format(self.net)
        self.num_images_test = len(self.imdb_test.image_index)
        self.all_boxes = [[[] for _ in range(self.num_images_test)]
                     for _ in range(self.imdb_test.num_classes)]
        self.output_dir = get_output_dir(self.imdb_test, save_name)
        dataset_test = roibatchLoader(roidb_test, ratio_list_test, ratio_index_test, self.batch_size_test, \
                                      self.imdb_test.num_classes, training=False, normalize=False)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size_test,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=True)

        self.iters_per_epoch = int(self.train_size / self.batch_size_train)

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def eval_frcnn(frcnn_extra, device, fasterRCNN, is_break=False):
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(frcnn_extra.output_dir, 'detections.pkl')
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    data_iter_test = iter(frcnn_extra.dataloader_test)
    for i in range(frcnn_extra.num_images_test):
        data_test = next(data_iter_test)
        im_data = data_test[0].to(device)
        im_info = data_test[1].to(device)
        gt_boxes = data_test[2].to(device)
        num_boxes = data_test[3].to(device)
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if frcnn_extra.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(frcnn_extra.imdb_test.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data_test[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        for j in range(1, frcnn_extra.imdb_test.num_classes):
            inds = torch.nonzero(scores[:, j] > frcnn_extra.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if frcnn_extra.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                frcnn_extra.all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                frcnn_extra.all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if frcnn_extra.max_per_image > 0:
            image_scores = np.hstack([frcnn_extra.all_boxes[j][i][:, -1]
                                      for j in range(1, frcnn_extra.imdb_test.num_classes)])
            if len(image_scores) > frcnn_extra.max_per_image:
                image_thresh = np.sort(image_scores)[-frcnn_extra.max_per_image]
                for j in range(1, frcnn_extra.imdb_test.num_classes):
                    keep = np.where(frcnn_extra.all_boxes[j][i][:, -1] >= image_thresh)[0]
                    frcnn_extra.all_boxes[j][i] = frcnn_extra.all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        if is_break:
            break
    ap = frcnn_extra.imdb_test.evaluate_detections(frcnn_extra.all_boxes, frcnn_extra.output_dir)
    return ap

def train_frcnn(frcnn_extra, device, fasterRCNN, optimizer, is_break=False):

    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    data_iter = iter(frcnn_extra.dataloader_train)
    for step in range(frcnn_extra.iters_per_epoch):
        data = next(data_iter)
        im_data = data[0].to(device)
        im_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        num_boxes = data[3].to(device)

        fasterRCNN.zero_grad()
        output = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = output

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if frcnn_extra.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        if is_break:
            break
    del output
    del loss
    return loss_temp