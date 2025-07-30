# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *

from torch.utils.data import ConcatDataset

def random_split(ls):
    length = len(ls)
    train_ls = ls[:ceil(length * 0.9)]
    val_ls = ls[ceil(length * 0.9):]
    return train_ls, val_ls

import os
import json
import nibabel as nib
from tqdm import tqdm 
CORRUPTED_FILE_PATH = "corrupted_files.json"  # 存放損壞檔案的 JSON

def filter_corrupted_datalist(datalist):
    """ 過濾損壞的 `.nii.gz` 檔案，並將結果儲存到 `corrupted_files.json` """
    
    # 讀取已存的損壞檔案列表，避免重複檢查
    if os.path.exists(CORRUPTED_FILE_PATH):
        with open(CORRUPTED_FILE_PATH, "r") as f:
            corrupted_files = set(json.load(f))  
    else:
        corrupted_files = set()

    valid_data = []
    new_corrupted = set()

    for item in tqdm(datalist, desc="🔍 檢查 .nii.gz 檔案是否損壞", unit="file"):
        nii_path = item["image"]

        # **如果檔案已標記為損壞，直接跳過**
        if nii_path in corrupted_files:
            print(f"🚨 已知損壞：{nii_path}，跳過")
            continue
        
        # 嘗試讀取 `.nii.gz`
        try:
            img = nib.load(nii_path)
            img.get_fdata()  # 確保能讀取數據
            valid_data.append(item)  # ✅ 正常檔案加入
        except Exception as e:
            print(f"❌ 損壞檔案：{nii_path}，錯誤: {e}")
            new_corrupted.add(nii_path)  # 記錄新發現的損壞檔案

    # ✅ 更新 `corrupted_files.json`
    if new_corrupted:
        corrupted_files.update(new_corrupted)
        with open(CORRUPTED_FILE_PATH, "w") as f:
            json.dump(list(corrupted_files), f, indent=4)
        print(f"📌 損壞檔案列表已更新，總共 {len(corrupted_files)} 個檔案")

    return valid_data  # **返回過濾後的 `datalist`**

def get_loader(args):
    splits0 = "20250327_pretrain_dataset_split.json"##
    # splits0 = "/pretrain_dataset_alldata.json"
    # splits0 = "all_abdominal.json"
    # splits1 = "/btcv.json"
    # splits2 = "/dataset_TCIAcovid19_0.json"
    # splits3 = "/dataset_LUNA16_0.json"
    # splits4 = "/stoic21.json"
    # splits5 = "/Totalsegmentator_dataset.json"
    # splits6 = "/flare23.json"
    splits6 = "/flare23_updated.json"##
    # splits7 = "/HNSCC.json"

    list_dir = "./jsons/"
    jsonlist1 = list_dir + splits0
    # jsonlist2 = list_dir + splits2
    # jsonlist3 = list_dir + splits3
    # jsonlist4 = list_dir + splits4
    # jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    # jsonlist7 = list_dir + splits7
    
    datadir0 = "./data/Abdominal"
    datadir1 = "./data/BTCV"
    datadir2 = "./data/TCIAcovid19"
    datadir3 = "./data/Luna16-jx"
    datadir4 = "./data/stoic21"
    datadir5 = "./data/Totalsegmentator_dataset"
    datadir6 = "./data/Flare23"
    datadir7 = "./data/HNSCC_convert_v1"

    num_workers = 12
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir0)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    # datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    # print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))

    # datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    # print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))
    # new_datalist3 = []
    # for item in datalist3:
    #     item_dict = {"image": item["image"]}
    #     new_datalist3.append(item_dict)

    # datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    # # datalist4, vallist4 = random_split(datalist4)
    # print("Dataset 4 TCIA Colon: number of data: {}".format(len(datalist4)))

    # datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    # # datalist5, vallist5 = random_split(datalist5)
    # print("Dataset 5 Totalsegmentator: number of data: {}".format(len(datalist5)))

    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    # datalist6, vallist6 = random_split(datalist6)
    print("Dataset 6 Flare23: number of data: {}".format(len(datalist6)))
    # **過濾掉損壞的 `.nii.gz`**
    # datalist6 = filter_corrupted_datalist(datalist6)
    # print("Dataset 6 Flare23: 過濾後數據數量: {}".format(len(datalist6)))

    # datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    # # datalist7, vallist7 = random_split(datalist7)
    # print("Dataset 7 HNSCC: number of data: {}".format(len(datalist7)))

    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir0)
    # vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    # vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)

    datalist = new_datalist1 + datalist6 # new_datalist1 + datalist2 + new_datalist3 + datalist4 + datalist5 + datalist6 + datalist7
    val_files = vallist1# + vallist2 + vallist3  # + vallist4 + vallist5 + vallist6 + vallist7
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    ###2D
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged,
        CropForegroundd, Resized, ToTensord
    )
    from utils.data_utils import VoCoAugmentation2D

    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),  # 加載 3D 資料
                                EnsureChannelFirstd(keys=["image"]),  # 確保 Channel 在第一維
                                Orientationd(keys=["image"], axcodes="RAS"),  # 統一方向
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
                                ),  # 縮放強度到 [0, 1]
                                CropForegroundd(keys="image", source_key="image"),  # 裁剪前景
                                Resized(keys="image", spatial_size=(384, 384, 96)),  # 3D 調整大小
                                # ToTensord(keys=["image"]),  # 轉為 Tensor
                                VoCoAugmentation2D(args, aug=False)  # 自定義的 2D 資料增強
    ])
    
    transform_npy = Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.uint8),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # scale 0~255 → 0~1
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys="image", source_key="image", select_fn=threshold),
        # Resized(keys="image", spatial_size=(384, 384, 96)),#Base Crop size = 4x4
        Resized(keys="image", spatial_size=(192, 192, 64)),#Base Crops size = 3x3
        VoCoAugmentation2D(args, aug=False),
    ])
    # Transform for HU data (e.g., Flare23 .nii.gz)
    transform_hu = Compose([
        LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # HU clipping to 0~1
        ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys="image", source_key="image", select_fn=threshold),
        # Resized(keys="image", spatial_size=(384, 384, 96)), #Base Crop size = 4x4
        Resized(keys="image", spatial_size=(192, 192, 64)),#Base Crops size = 3x3
        VoCoAugmentation2D(args, aug=False),
    ])
    ###

    if args.cache_dataset: 
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms,
                                cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset") # Run this

        print("Concat Data1&Data6")
        dataset_npy = Dataset(data=new_datalist1, transform=transform_npy)
        # train_ds = Dataset(data=new_datalist1, transform=transform_npy)
        dataset_hu = Dataset(data=datalist6, transform=transform_hu)
        # 合併成一個 dataset
        train_ds = ConcatDataset([dataset_npy, dataset_hu])
        
        # train_ds = Dataset(data=datalist, transform=train_transforms)


        # train_ds = PersistentDataset(data=datalist,
        #                              transform=train_transforms,
        #                              pickle_protocol=pickle.HIGHEST_PROTOCOL,
        #                              cache_dir='/data/linshan/cache/10k')



    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    print(f"num_workers = {num_workers}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=False,
        drop_last=True, pin_memory=True
    )
    return train_loader

import matplotlib.pyplot as plt


def threshold(x):
    # threshold at 0
    return x > 0.3


def visualize_crops_and_labels(random_crop, base_crops, labels, output_path="visualization.png"):
    """
    可視化同一個切片的 16 塊 base crops 和 random crop，並顯示相似度，將結果保存為圖片。
    """
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # 顯示 random crop
    axes[0, 0].imshow(random_crop[0, :, :], cmap="gray")
    axes[0, 0].set_title("Random Crop")
    axes[0, 0].axis("off")

    # 顯示 16 塊 base crops
    for i in range(16):
        row, col = divmod(i, 4)
        base_crop = base_crops[i, 0, :, :]  # [1, H, W]
        similarity = labels[i].item()  # label 值

        ax = axes[row + 1, col]  # 第 2 行開始是 base crops
        ax.imshow(base_crop, cmap="gray")
        ax.set_title(f"Base {i}\nLabel: {similarity:.2f}")
        ax.axis("off")

    # 隱藏多餘的 subplot
    for ax in axes[0][1:]:
        ax.axis("off")

    # 保存圖片
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close(fig)

import torch
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandShiftIntensityd, ToTensord, Resized

class VoCoAugmentation2D:
    def __init__(self, args, aug):
        self.args = args
        self.aug = aug

    def __call__(self, x_in):
        # print(f"x_in keys: {x_in.keys()}")
        # print(f"x_in['image'] shape: {x_in['image'].shape}")#  [C, H, W, D]

        # Step 1: 3D 裁剪
        crops_trans = get_crop_transform(roi_small=self.args.roi_x, aug=self.aug)
        # vanilla_trans, positions = new_get_vanilla_transform(num=self.args.sw_batch_size, roi_small=self.args.roi_x, aug=self.aug)
        vanilla_trans, labels = get_vanilla_transform(num=self.args.sw_batch_size,roi_small=self.args.roi_x, aug=self.aug)
        # for i, trans in enumerate(crops_trans):
        #     result = trans(x_in)  # Apply transformation
        #     print(f"Crop {i} shape: {result['image'].shape}")
        imgs_3d = [trans(x_in)["image"] for trans in vanilla_trans]
        crops_3d = [trans(x_in)["image"] for trans in crops_trans]

        # Step 2: 以切片為單位處理資料
        imgs_slices, crops_slices, labels_slices = self.slice_and_reorganize(imgs_3d, crops_3d, labels)

        # print(f"Processed images shape: {imgs_slices.shape}")
        # print(f"Processed crops shape: {crops_slices.shape}")
        # print(f"Processed labels shape: {labels_slices.shape}")

        return imgs_slices, labels_slices, crops_slices

    # def slice_and_reorganize(self, imgs_3d, crops_3d, labels, interval=3, visualize=True):
    #     """
    #     將 3D 資料切成 2D 並按切片重新組織。
    #     """
    #     num_slices = imgs_3d[0].shape[-1]  # Z 軸的深度 [C, H, W, D]
    #     slices_imgs = []  # 每片切片的 random crops
    #     slices_crops = []  # 每片切片的 base crops
    #     slices_labels = []  # 每片切片的標籤

    #     # print(f"img_3d[0] shape = {imgs_3d[0].shape}")
    #     # print(f"crops_3d[0] shape = {crops_3d[0].shape}")
    #     # print(f"Original labels shape = {labels.shape}")  # [2, 16]
    #     labels = torch.tensor(labels, dtype=torch.float32)# nparray -> tensor
    #     for z_idx in range(0, num_slices, interval):
    #         # 提取每片切片的 random 和 base crops
    #         random_slices = torch.stack([img_3d[:, :, :, z_idx] for img_3d in imgs_3d])  # 2 個 random crops
    #         base_slices = torch.stack([crop_3d[:, :, :, z_idx] for crop_3d in crops_3d])  # 16 個 base crops

    #         # 計算每片切片的標籤
    #         # random_crop_labels = self.calculate_labels(random_slices, base_slices, positions, roi=self.args.roi_x)
            
    #         slices_imgs.append(random_slices)  # [2, 1, 64, 64]
    #         slices_crops.append(base_slices)  # [16, 1, 64, 64]
    #         slices_labels.append(labels)  # [2, 16]

    #         # # 可視化
    #         # if visualize and z_idx == 16:  # 僅可視化第一個 Z 切片
    #         #     random_crop = random_slices[0]  # 第0個 random crop
    #         #     base_crops = base_slices  # 全部 16 塊 base crops
    #         #     labels_vis = labels[0,:]  # 第0個 random crop labels

    #         #     # 呼叫可視化函數
    #         #     visualize_crops_and_labels(random_crop, base_crops, labels_vis, output_path=f"./abdominal_slice/augment_{z_idx}.png")
    #         #     # visualize_crops_and_labels(random_crop, base_crops, labels_vis, output_path=f"./abdominal_slice/abdominal_slice_{z_idx}.png")

    #         #     breakpoint()

    #     # 組織成 Tensor
    #     slices_imgs = torch.stack(slices_imgs)  # [64 (slices), 2 (random), 1, 64, 64]
    #     slices_crops = torch.stack(slices_crops)  # [64 (slices), 16 (base), 1, 64, 64]
    #     slices_labels = torch.stack(slices_labels)  # [64 (slices), 2, 16]

    #     return slices_imgs, slices_crops, slices_labels
    
    def slice_and_reorganize(self, imgs_3d, crops_3d, labels, target_slices=32, visualize=True):
        """
        將 3D 資料切成指定數量的 2D 並按切片重新組織。
        """
        num_slices = imgs_3d[0].shape[-1]  # Z 軸的深度，假設為 64
        assert target_slices <= num_slices, "target_slices 不能大於原始切片數"

        # 等距地選擇切片 index，例如 0~63 中挑 48 張
        selected_z = torch.linspace(0, num_slices - 1, steps=target_slices).long()

        slices_imgs = []
        slices_crops = []
        slices_labels = []

        labels = torch.tensor(labels, dtype=torch.float32)  # nparray -> tensor

        for z_idx in selected_z:
            random_slices = torch.stack([img_3d[:, :, :, z_idx] for img_3d in imgs_3d])
            base_slices = torch.stack([crop_3d[:, :, :, z_idx] for crop_3d in crops_3d])

            slices_imgs.append(random_slices)
            slices_crops.append(base_slices)
            slices_labels.append(labels)

        slices_imgs = torch.stack(slices_imgs)      # [target_slices, 2, 1, 64, 64]
        slices_crops = torch.stack(slices_crops)    # [target_slices, 16, 1, 64, 64]
        slices_labels = torch.stack(slices_labels)  # [target_slices, 2, 16]

        return slices_imgs, slices_crops, slices_labels


    # def calculate_labels(self, random_slices, base_slices, positions, roi):
    #     """
    #     計算每片切片的 random 和 base crops 的重疊比例標籤。
    #     """
    #     num_random = random_slices.size(0)  # 2
    #     num_base = base_slices.size(0)  # 16
    #     labels = torch.zeros(1, num_random, num_base)

    #     random_positions = positions[:num_random]  # 隨機裁剪中心座標
    #     base_positions = self.get_base_positions(roi)  # 基礎裁剪中心座標

    #     for i, rand_pos in enumerate(random_positions):
    #         for j, base_pos in enumerate(base_positions):
    #             labels[0, i, j] = self.calculate_overlap(rand_pos, base_pos, roi)

    #     return labels

    # def get_base_positions(self, roi):
    #     """
    #     返回基礎裁剪的中心座標。
    #     """
    #     num_crops = 4  # 4x4 的 base crops
    #     base_positions = []
    #     for i in range(num_crops):
    #         for j in range(num_crops):
    #             center_x = (i + 0.5) * roi
    #             center_y = (j + 0.5) * roi
    #             base_positions.append((center_x, center_y))
    #     return base_positions

    # def calculate_overlap(self, rand_pos, base_pos, roi):
    #     """
    #     計算隨機裁剪和基礎裁剪的重疊比例。
    #     """
    #     rand_x, rand_y = rand_pos
    #     base_x, base_y = base_pos

    #     half_roi = roi / 2
    #     rand_min_x, rand_max_x = rand_x - half_roi, rand_x + half_roi
    #     rand_min_y, rand_max_y = rand_y - half_roi, rand_y + half_roi

    #     base_min_x, base_max_x = base_x - half_roi, base_x + half_roi
    #     base_min_y, base_max_y = base_y - half_roi, base_y + half_roi

    #     # 計算重疊區域
    #     overlap_x = max(0, min(rand_max_x, base_max_x) - max(rand_min_x, base_min_x))
    #     overlap_y = max(0, min(rand_max_y, base_max_y) - max(rand_min_y, base_min_y))
    #     overlap_area = overlap_x * overlap_y

    #     # 正規化重疊比例
    #     total_area = roi * roi
    #     return overlap_area / total_area

def get_vanilla_transform(num=2, num_crops=3, roi_small=64, roi=64, max_roi=192, aug=True):#(num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=True):#
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(roi=roi,
                                                       max_roi=max_roi,
                                                       num_crops=num_crops)
        if aug:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                ToTensord(keys=["image"])])
        else:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                ToTensord(keys=["image"])])

        vanilla_trans.append(trans)
        labels.append(label)

    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)

    return vanilla_trans, labels


def get_crop_transform(num_crops=3, roi_small=64, roi=64, aug=True):#(num_crops=4, roi_small=64, roi=96, aug=True): #
    voco_trans = []
    # not symmetric at axis x !!!
    for i in range(num_crops):
        for j in range(num_crops):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2

            if aug:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    Resized(keys=["image"], mode="bilinear", align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                    ToTensord(keys=["image"])],
                )
            else:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    ToTensord(keys=["image"])],
                )

            voco_trans.append(trans)
    return voco_trans


def get_position_label(roi=64, base_roi=64, max_roi=192, num_crops=3): #(roi=96, base_roi=96, max_roi=384, num_crops=4):# 
    half = roi // 2
    center_x, center_y = np.random.randint(low=half, high=max_roi - half), \
        np.random.randint(low=half, high=max_roi - half)
    # center_x, center_y = np.random.randint(low=half, high=half+1), \
    #     np.random.randint(low=half, high=half+1)
    # center_x, center_y = roi + half, roi + half
    # print(center_x, center_y)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for i in range(num_crops):
        for j in range(num_crops):
            crop_x_min, crop_x_max = i * base_roi, (i + 1) * base_roi
            crop_y_min, crop_y_max = j * base_roi, (j + 1) * base_roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)

    return center_x, center_y, labels


if __name__ == '__main__':
    # center_x, center_y, labels = get_position_label()
    # print(center_x, center_y, labels)
    print("--main function--")