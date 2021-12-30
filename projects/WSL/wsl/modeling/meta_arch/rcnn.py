# -*- coding: utf-8 -*-

import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..postprocessing import detector_postprocess
from skimage import measure
import os
import cv2

import pdb

__all__ = ["GeneralizedRCNNWSL", "ProposalNetworkWSL"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWSL(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        has_cpg: bool,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.has_cpg = has_cpg

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        cls.visualize_pgta = cfg.MODEL.PGTA.VISUALIZE
        cls.visualize_pua = cfg.MODEL.PUA.VISUALIZE
        if cls.visualize_pgta or cls.visualize_pua:
            cls.image_cnt = 0
            cls.output_dir_pgta = cfg.MODEL.WEIGHTS.split(".")[0] + "_pgta"
            cls.output_dir_pua = cfg.MODEL.WEIGHTS.split(".")[0] + "_pua"
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "has_cpg": True
            if "CSC" in cfg.MODEL.ROI_HEADS.NAME or "WSJDS" in cfg.MODEL.ROI_HEADS.NAME
            # or "UWSODROIHeads" in cfg.MODEL.ROI_HEADS.NAME
            else False,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.has_cpg:
            images.tensor.requires_grad = True

        features = self.backbone(images.tensor)
        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, gt_instances)
            for i, p in enumerate(proposals):
                # corner case
                if len(p) == 0:
                    device = images.tensor.device
                    image_size = images.image_sizes[0]

                    b_new = Boxes(
                        torch.tensor(
                            [[0.0, 0.0, image_size[1] - 1, image_size[0] - 1]],
                            device=device,
                            dtype=torch.float,
                        )
                    )
                    s_new = torch.tensor([1.0], device=device, dtype=torch.float)

                    p = Instances(image_size)
                    p.proposal_boxes = b_new
                    p.objectness_logits = s_new
                    p.level_ids = [0]
                    proposals[i] = p
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.proposal_generator:
            proposal_losses = self.proposal_generator.get_losses(self.roi_heads.proposal_targets)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.has_cpg:
            images.tensor.requires_grad = False
            images.tensor.detach()

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            if self.visualize_pgta:
                results, _, all_scores, all_boxes, pgta_data = self.roi_heads(images, features, proposals, None)
                self.draw_pgta(images, pgta_data)
            if self.visualize_pua:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                results, _, all_scores, all_boxes, pua_data = self.roi_heads(images, features, proposals, gt_instances)
                self.draw_pua(images, pua_data)
            else:
                results, _, all_scores, all_boxes = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results, all_scores, all_boxes = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNNWSL._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results, all_scores, all_boxes

    def draw_pgta(self, images, pgta_data):
        def _draw_heatmap(image, heatmap):
            heatmap = heatmap - np.min(heatmap)
            heatmap = heatmap / np.max(heatmap)
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)   # 将热力图分散到rgb通道
            img_with_hm = np.float32(heatmap) / 255 + np.float32(image) / 255
            img_with_hm = img_with_hm / np.max(img_with_hm)
            img_with_hm = np.uint8(255 * img_with_hm)
            return img_with_hm
        
        def _draw_rectangle(image, heatmap):
            _area_thres = (heatmap.shape[0] * heatmap.shape[1]) / 100
            _num_stride = 10
            _start = heatmap.mean()
            _stride = (heatmap.max()-_start) / _num_stride
            for i in range(_num_stride):
                thres = i * _stride + _start
                mask = np.where(heatmap > thres, 1, 0)
                labels = measure.label(mask, connectivity=2)
                regions = measure.regionprops(labels)
                for region in regions:
                    if region.area < _area_thres:
                        continue
                    y1, x1, y2, x2 = region.bbox
                    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 255, 0), 1)
            return image

        if not os.path.exists(self.output_dir_pgta):
            os.makedirs(self.output_dir_pgta)

        im = images.tensor[0, ...].clone().detach().cpu().numpy()
        im_w, im_h = images.image_sizes[0][1], images.image_sizes[0][0]
        im = im.transpose((1, 2, 0))
        pixel_means = [102.9801, 115.9465, 122.7717]
        im += pixel_means
        im = np.ascontiguousarray(im)
        im = im.astype(np.uint8)
        
        source_map = pgta_data['source_map'].detach()
        pgt_map = pgta_data['pgt_map'].detach()

        # source_map = F.interpolate(source_map, (im_h, im_w)).squeeze().cpu().numpy()
        # source_hm = _draw_heatmap(im, source_map)
        # cv2.imwrite(os.path.join(self.output_dir_pgta, 'source_{}.jpg'.format(self.image_cnt)), source_hm)
        
        pgt_map = F.interpolate(pgt_map, (im_h, im_w)).squeeze().cpu().numpy()
        proposal_img = _draw_rectangle(im, pgt_map)
        cv2.imwrite(os.path.join(self.output_dir_pgta, 'bbox_{}.jpg'.format(self.image_cnt)), proposal_img)
        pgt_hm = _draw_heatmap(im, pgt_map)
        cv2.imwrite(os.path.join(self.output_dir_pgta, 'pgt_{}.jpg'.format(self.image_cnt)), pgt_hm)
        self.image_cnt += 1

    def draw_pua(self, images, pua_data):
        if not os.path.exists(self.output_dir_pua):
            os.makedirs(self.output_dir_pua)

        im = images.tensor[0, ...].clone().detach().cpu().numpy()
        im_w, im_h = images.image_sizes[0][1], images.image_sizes[0][0]
        im = im.transpose((1, 2, 0))
        pixel_means = [102.9801, 115.9465, 122.7717]
        im += pixel_means
        im = np.ascontiguousarray(im)
        im = im.astype(np.uint8)

        box_weight = pua_data['box_weight'].cpu().numpy()
        box_pos = pua_data['box_pos'].cpu().numpy()   # [x1, y1, x2, y2]
        box_logit = pua_data['box_logit'].cpu().numpy()

        # box_weight
        # for x1, y1, x2, y2 in box_pos[np.where(box_weight > 0.9)[0]]:
        #     cv2.rectangle(im, (int(x1), int(y1)), (int(x2)-1, int(y2)-1), (0, 0, 255), 1)   # red
        # for x1, y1, x2, y2 in box_pos[np.where(box_weight < 0.1)[0]]:
        #     cv2.rectangle(im, (int(x1), int(y1)), (int(x2)-1, int(y2)-1), (255, 0, 0), 1)   # blue

        # box_logit
        # for x1, y1, x2, y2 in box_pos[np.where(box_logit < 1.)[0]]:
        #     cv2.rectangle(im, (int(x1), int(y1)), (int(x2)-1, int(y2)-1), (0, 255, 0), 1)   # green

        # box_logit_weight
        box_lw = box_logit * box_weight
        box_lw_min, box_lw_max = np.min(box_lw), np.max(box_lw)
        box_lw = (box_lw - box_lw_min) / (box_lw_max - box_lw_min)
        for x1, y1, x2, y2 in box_pos[np.where(box_lw > 0.9)[0]]:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2)-1, int(y2)-1), (0, 255, 255), 1)   # yellow

        cv2.imwrite(os.path.join(self.output_dir_pua, 'pua_{}_oicr_gt_softmax.jpg'.format(self.image_cnt)), im)
        self.image_cnt += 1
        if self.image_cnt > 20:
            exit(0)
        # pdb.set_trace()

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetworkWSL(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNNWSL.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
