###############################################################################
##  (C) MIT License (https://github.com/ylabbe/cosypose/blob/master/LICENSE) ##
##  Code to evaluate images on the cosypose program for 6D-Pose-Estimation   ##
##  This code is composed from many files from the lib-folder.               ##
##                                                                           ##
##  Author: gezp (https://github.com/gezp) [Composition of code snippets]    ##
##  Edited by: datboi223                                                     ##
###############################################################################

# https://github.com/ylabbe/cosypose/issues/17 <- Look at the code

import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import threading
lock = threading.Lock()

# ROS Imports
import roslib, rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset

# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.evaluation.pred_runner.bop_predictions import BopPredictionRunner

from cosypose.utils.distributed import get_tmp_dir, get_rank
from cosypose.utils.distributed import init_distributed_mode

from cosypose.config import EXP_DIR, RESULTS_DIR

# From Notebook
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.singleview import render_prediction_wrt_camera
from cosypose.visualization.plotter import Plotter
from bokeh.io import export_png
from bokeh.plotting import gridplot

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(coarse_run_id, refiner_run_id=None, n_workers=8):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    #object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def getModel(model_type):
    #load models
    if model_type == 'tless':
        detector_run_id='detector-bop-tless-pbr--873074'
        coarse_run_id='coarse-bop-tless-pbr--506801'
        refiner_run_id='refiner-bop-tless-pbr--233420'
        detector = load_detector(detector_run_id)
        pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id,refiner_run_id=refiner_run_id,n_workers=4)
        return detector,pose_predictor
    
    if model_type == 'ycb':
        detector_run_id='detector-bop-ycbv-pbr--970850'
        coarse_run_id='coarse-bop-ycbv-pbr--724183'
        refiner_run_id='refiner-bop-ycbv-pbr--604090'
        detector = load_detector(detector_run_id)
        pose_predictor, mesh_db = load_pose_models(coarse_run_id=coarse_run_id,refiner_run_id=refiner_run_id,n_workers=4)
        return detector,pose_predictor


def inference(detector,pose_predictor,image,camera_k):
    #[1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    #[1,3,3]
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    #2D detector 
    #print("start detect object.")
    box_detections = detector.get_detections(images=images, one_instance_per_class=False, 
                    detection_th=0.8,output_masks=False, mask_th=0.9)
    #pose esitimition
    if len(box_detections) == 0:
        return None
    #print("start estimate pose.")
    final_preds, all_preds=pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                        n_coarse_iterations=1,n_refiner_iterations=4)
    #print("inference successfully.")
    #result: this_batch_detections, final_preds
    return final_preds.cpu(), box_detections


class CosyPose:
    def __init__(self, model_type):
        self.nr = 0
        self.im = None
        self.camera_info = None
        self.model_type = model_type
        self.detector, self.pose_predictor = getModel(self.model_type)
        self.bridge = CvBridge()


        self.nr = 0 # for indexing purposes

        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=1) # 10
        self.camera_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo, queue_size=1) # 10

        queue_size = 5
        slop_seconds = 0.1
        sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, 
                                                            self.camera_sub],
                                                            queue_size, slop_seconds)
        sync.registerCallback(self.callback)
        print('Waiting for Messages')


        self.tf_pub = None
        

    def callback(self, rgb_msg, camera_msg):
        # get the rgb-data
        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        except CvBridgeError as e:
            print(e)

        # get the Camera-Info-Message
        try:
            camera_message = camera_msg
        except Exception as e:
            print(e)

        with lock:
            self.im = cv_rgb_image.copy()
            self.camera_info = camera_message


    def process_data(self):
        with lock:
            if self.im is None:
                return 
            im_color = self.im.copy()
            camera_info = self.camera_info

        print('\n#############################################################################')
        print('Recieved Message-Nr[{}]'.format(self.nr))
        H, W, _ = im_color.shape
        input_dim = (W, H)
        camera_K = np.array(camera_info.K).reshape(3, 3)

        pred, detections = inference(self.detector, self.pose_predictor, im_color, camera_K)
        # poses,poses_input,K_crop,boxes_rend,boxes_crop
        print("num of pred:",len(pred))
        for i in range(len(pred)):
            print("object ", i, ":", pred.infos.iloc[i].label, "------\n  pose:",
                  pred.poses[i].numpy(), "\n  detection score:", pred.infos.iloc[i].score)


        self.nr += 1




def main():

    model_type = 'ycb'
    poseEstimator = CosyPose(model_type)
    try:
        while not rospy.is_shutdown():
            poseEstimator.process_data()
    except rospy.ROSInterruptException:
        exit()

    exit()    

if __name__ == '__main__':
    rospy.init_node('cosypose_ros', anonymous=True)
    main()