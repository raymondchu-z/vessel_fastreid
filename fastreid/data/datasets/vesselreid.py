# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import os
import re
import warnings

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VesselReid(ImageDataset):
    dataset_dir = ''
    dataset_name = "vesselreid"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        # print(root)
        root = '/home/zlm/research/fast-reid/datasets'
        self.root = root
        # print(self.root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # print(self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'vesselreid')
        # print(data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'train_all')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        # self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # print(self.train_dir)
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)


        super(VesselReid, self).__init__(train, query, gallery, **kwargs)


    def show_files(self, path, all_files):
        # 首先遍历当前目录所有文件及文件夹
        file_list = os.listdir(path)
        # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
        for file in file_list:
            # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
            cur_path = os.path.join(path, file)
            # 判断是否是文件夹
            if os.path.isdir(cur_path):
                self.show_files(cur_path, all_files)
            else:
                if(file.endswith('jpg')):
                    all_files.append(osp.join(cur_path))

        return all_files

    def process_dir(self, dir_path, is_train=True):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths = []
        # for dirpath, dirs, rootfiles in os.walk(dir_path):
        #     for dir in dirs:
        #         img_paths.append(glob.glob(osp.join(dirpath, dir, '*.jpg')))
        #         for pic in rootfiles:
        #             img_paths.append(osp.join(dirpath, dir, pic))
        img_paths = self.show_files(dir_path, [])
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            # print(img_path)
            IMO = img_path.split('/')[-2]
            # pid, camid = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue  # junk images are just ignored

            # camid -= 1  # index starts from 0
            if is_train:
                pid = str(IMO)
                camid = str(0)
            else:
                pid = int(IMO)
                if(img_path.split('/')[-3]=='gallery'):
                    camid = 1
                else:
                    camid = 2
            data.append((img_path, pid, camid))
            


        return data

if __name__ == "__main__":
    VesselReid()