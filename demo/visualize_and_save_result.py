# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import os
import pickle
import numpy as np
import torch
import tqdm
from torch.backends import cudnn


sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer


cudnn.benchmark = True
logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        # default='VesselReid',
        default='Market1501',
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=10,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def save_features(output, feats, pids, camids):
    if not os.path.exists(output):
        os.makedirs(output)
    features = {
        "feats": np.asarray(feats),
        "pids": np.asarray(pids),
        "camids": np.asarray(camids),
    }
    with open(os.path.join(output, "features.pickle"), "wb") as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load_features(path):
    with open(path, 'rb') as handle: res = pickle.load(handle)
    return res

if __name__ == '__main__':
    args = get_parser().parse_args()
    logger = setup_logger()
    cfg = setup_cfg(args)
    test_loader, num_query = build_reid_test_loader(cfg, args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    if(os.path.exists(os.path.join(args.output, "features.pickle"))):
        features = load_features(os.path.join(args.output, "features.pickle"))
        logger.info("features load at " + args.output,)
        feats = features['feats']
        pids = features['pids']
        camids = features['camids']
        feats = torch.from_numpy(feats)
    else:
        logger.info("Start extracting image features")# 10min
        feats = []
        pids = []
        camids = []
        for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)

        feats = torch.cat(feats, dim=0)
        save_features(args.output, feats, pids, camids)
        print("features saved at " + args.output,)
    
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query]).astype(np.int32)
    g_pids = np.asarray(pids[num_query:]).astype(np.int32)
    q_camids = np.asarray(camids[:num_query]).astype(np.int8)
    g_camids = np.asarray(camids[num_query:]).astype(np.int8)

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()

    logger.info("Computing APs for all query images ...")# 10min
    # cmc, all_ap, all_inp = evaluate_rank(distmat, q_feat, g_feat, q_pids, g_pids, q_camids, g_camids)
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids) #修改为
    visualizer = Visualizer(test_loader.dataset)
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Saving ROC curve and distribution...")
    visualizer.save_roc_curve(args.output)
    # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)

    

    logger.info("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
