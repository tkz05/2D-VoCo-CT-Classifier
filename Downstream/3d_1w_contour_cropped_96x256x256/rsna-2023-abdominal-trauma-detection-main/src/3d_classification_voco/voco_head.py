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

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
import timm


class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.proj_head = projection_head(in_dim=1152, hidden_dim=2048, out_dim=2048)

    def forward_encs(self, encs):
        b = encs[0].size()[0]
        outs = []
        for enc in encs:
            out = F.adaptive_avg_pool3d(enc, (1, 1, 1))
            outs.append(out.view(b, -1))
        outs = torch.cat(outs, dim=1)
        return outs

    def forward(self, x_in):
        b = x_in.size()[0]
        hidden_states_out = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]

        # for enc in encs:
        #     print(enc.shape)

        out = self.forward_encs(encs)
        out = self.proj_head(out.view(b, -1))
        return out

# 定義新的投影頭，用於特徵向量的映射
class ProjectionHead(nn.Module):
    """
    投影頭將 backbone 提取的特徵進一步映射到特定維度（默認 2048），以進行教師-學生模型的對比學習。
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super(ProjectionHead, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """
        輸入特徵向量，經過三層線性層和非線性變換後，返回最終投影向量。
        """
        x = self.layer1(x)  # 第一層映射與歸一化
        x = self.layer2(x)  # 第二層映射與歸一化
        x = self.layer3(x)  # 最終投影
        return x


# 修改過的 VoCoHead，適配 efficientnetv2_rw_t 作為 backbone
class VoCoHead(nn.Module):
    """
    VoCoHead 用於訓練一個教師-學生模型，學生模型（student）學習輸出與教師模型（teacher）保持一致。
    """
    def __init__(self, args):
        super(VoCoHead, self).__init__()
        
        # 定義學生模型的 backbone（efficientnetv2_rw_t）
        self.student = timm.create_model(
            "efficientnetv2_rw_t",  # 使用 EfficientNet v2 作為 backbone
            pretrained=True,  # 不使用預訓練權重
            in_chans=args.in_channels,  # 輸入通道數，與下游模型一致（如 CT 圖像為 1）
            drop_rate=0.1,  # Dropout 機率
            drop_path_rate=0.0  # DropPath 機率
        
        )
        print("pretrained=True,")
        # 定義教師模型的 backbone，與學生模型一致
        self.teacher = timm.create_model(
            "efficientnetv2_rw_t",
            pretrained=True,
            in_chans=args.in_channels,
            drop_rate=0.1,
            drop_path_rate=0.0
        )
        print(f"Student Backbone Output Features: {self.student.num_features}")
        # 投影頭（Projection Head），將 backbone 特徵映射到對比學習的向量空間
        self.student_proj_head = ProjectionHead(in_dim=self.student.num_features)  # 使用 EfficientNet 的特徵維度
        self.teacher_proj_head = ProjectionHead(in_dim=self.teacher.num_features)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        """
        使用指數移動平均（EMA）更新教師模型的參數。
        """
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def forward(self, img, crops, labels):
        """
        前向傳播方法，包含形狀檢查的打印語句。

        Args:
            img (torch.Tensor): 隨機裁剪影像，形狀為 [num_slices, num_random_crops, C, H, W]。
            crops (torch.Tensor): 基礎裁剪影像，形狀為 [num_slices, num_base_crops, C, H, W]。
            labels (torch.Tensor): 標籤，形狀為 [num_slices, num_random_crops, num_base_crops]。

        Returns:
            tuple: 包含正損失、負損失和基礎裁剪正則化損失的元組。
        """
        # 打印輸入形狀
        # print(f"Input img shape: {img.shape}")
        # print(f"Input crops shape: {crops.shape}")
        # print(f"Input labels shape: {labels.shape}")

        num_slices, num_random_crops, _, _, _ = img.size()
        _, num_base_crops, _, _, _ = crops.size()

        img, crops = img.as_tensor(), crops.as_tensor()
        # 將 num_slices 和 num_random_crops 或 num_base_crops 合併為批量大小
        img = img.view(-1, *img.shape[2:])  # [num_slices * num_random_crops, C, H, W]
        crops = crops.view(-1, *crops.shape[2:])  # [num_slices * num_base_crops, C, H, W]
        # print(f"Reshaped img shape: {img.shape}")
        # print(f"Reshaped crops shape: {crops.shape}")

        # 合併影像輸入，並通過學生和教師模型
        inputs = torch.cat([img, crops], dim=0)  # [num_slices * (num_random_crops + num_base_crops), C, H, W]
        # print(f"Combined input shape: {inputs.shape}")
        student_features = self.student.forward_features(inputs)
        # print(f"Student features shape before projection: {student_features.shape}")
        self._EMA_update_encoder_teacher()
        with torch.no_grad():
            teacher_features = self.teacher.forward_features(inputs).detach()
        # print(f"Teacher features shape before projection: {teacher_features.shape}")

        # 使用全局平均池化對特徵降維，統一形狀為 [batch_size, features]
        def _reduce_features(features):
            batch_size = features.size(0)
            reduced_features = F.adaptive_avg_pool2d(features, (1, 1)).view(batch_size, -1)
            return reduced_features

        student_features_reduced = _reduce_features(student_features)
        teacher_features_reduced = _reduce_features(teacher_features)
        # print(f"Student features reduced shape: {student_features_reduced.shape}")
        # print(f"Teacher features reduced shape: {teacher_features_reduced.shape}")

        # 投影頭處理降維後的特徵
        student_img_features = self.student_proj_head(student_features_reduced[:len(img)])
        student_crop_features = self.student_proj_head(student_features_reduced[len(img):])
        teacher_img_features = self.teacher_proj_head(teacher_features_reduced[:len(img)])
        teacher_crop_features = self.teacher_proj_head(teacher_features_reduced[len(img):])
        # print(f"Student img features after projection: {student_img_features.shape}")
        # print(f"Student crop features after projection: {student_crop_features.shape}")
        # print(f"Teacher img features after projection: {teacher_img_features.shape}")
        # print(f"Teacher crop features after projection: {teacher_crop_features.shape}")

        # 初始化損失
        pos_loss, neg_loss, reg_loss = 0.0, 0.0, 0.0

        # 按切片進行對比學習
        for slice_idx in range(num_slices):
            # print(f"Processing slice {slice_idx + 1}/{num_slices}")
            # 提取當前切片的隨機裁剪和基礎裁剪特徵
            slice_img_features = student_img_features[slice_idx * num_random_crops:(slice_idx + 1) * num_random_crops]
            slice_crop_features = student_crop_features[slice_idx * num_base_crops:(slice_idx + 1) * num_base_crops]
            teacher_slice_img_features = teacher_img_features[slice_idx * num_random_crops:(slice_idx + 1) * num_random_crops]
            teacher_slice_crop_features = teacher_crop_features[slice_idx * num_base_crops:(slice_idx + 1) * num_base_crops]
            # print(f"Slice {slice_idx}: student img features shape: {slice_img_features.shape}")
            # print(f"Slice {slice_idx}: student crop features shape: {slice_crop_features.shape}")
            # print(f"Slice {slice_idx}: teacher img features shape: {teacher_slice_img_features.shape}")
            # print(f"Slice {slice_idx}: teacher crop features shape: {teacher_slice_crop_features.shape}")

            # 計算對比學習損失
            logits1 = self._contrastive_logits(slice_img_features, teacher_slice_crop_features)
            logits2 = self._contrastive_logits(teacher_slice_img_features, slice_crop_features)
            logits = (logits1 + logits2) * 0.5

            # 取出對應的標籤
            slice_labels = labels[slice_idx]  # [num_random_crops, num_base_crops]
            # print(f"Slice {slice_idx}: logits shape: {logits.shape}")
            # print(f"Slice {slice_idx}: labels shape: {slice_labels.shape}")

            # 計算損失
            slice_pos_loss, slice_neg_loss = self._contrastive_loss(slice_labels, logits)
            pos_loss += slice_pos_loss
            neg_loss += slice_neg_loss

            # 基礎裁剪特徵的正則化損失
            slice_reg_loss = self._regularization_loss(slice_crop_features)
            reg_loss += slice_reg_loss

        # 平均損失
        pos_loss /= num_slices
        neg_loss /= num_slices
        reg_loss /= num_slices

        # print(f"Final pos_loss: {pos_loss}, neg_loss: {neg_loss}, reg_loss: {reg_loss}")
        return pos_loss, neg_loss, reg_loss


    def _contrastive_logits(self, img_features, crop_features):
        """
        計算隨機裁剪與基礎裁剪特徵之間的相似度。

        Args:
            img_features (torch.Tensor): 隨機裁剪特徵，形狀為 [num_random_crops, feature_dim]。
            crop_features (torch.Tensor): 基礎裁剪特徵，形狀為 [num_base_crops, feature_dim]。

        Returns:
            torch.Tensor: 相似度 logits，形狀為 [num_random_crops, num_base_crops]。
        """
        logits = F.cosine_similarity(img_features.unsqueeze(1), crop_features.unsqueeze(0), dim=-1)
        return F.relu(logits)

    def _contrastive_loss(self, labels, logits):
        """
        計算對比學習損失。

        Args:
            labels (torch.Tensor): 標籤，形狀為 [num_random_crops, num_base_crops]。
            logits (torch.Tensor): 預測 logits，形狀為 [num_random_crops, num_base_crops]。

        Returns:
            tuple: 包含正損失和負損失的元組。
        """
        pos_dis = torch.abs(labels - logits)
        pos_loss = -labels * torch.log(1 - pos_dis + 1e-6)
        pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

        neg_mask = (labels == 0).float()
        neg_loss = neg_mask * (logits ** 2)
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-6)

        return pos_loss, neg_loss

    def _regularization_loss(self, crop_features):
        """
        計算基礎裁剪特徵的正則化損失。

        Args:
            crop_features (torch.Tensor): 基礎裁剪特徵，形狀為 [num_base_crops, feature_dim]。

        Returns:
            torch.Tensor: 正則化損失。
        """
        k, _ = crop_features.size()
        loss = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                similarity = F.cosine_similarity(crop_features[i], crop_features[j], dim=0)
                similarity = F.relu(similarity)  # 使用 ReLU 避免負相似度
                loss += similarity ** 2
        return loss / (k * (k - 1) / 2)


# # 修改過的 VoCoHead，適配 efficientnetv2_rw_t 作為 backbone
# class VoCoHead(nn.Module):
#     """
#     VoCoHead 用於訓練一個教師-學生模型，學生模型（student）學習輸出與教師模型（teacher）保持一致。
#     """
#     def __init__(self, args):
#         super(VoCoHead, self).__init__()
        
#         # 定義學生模型的 backbone（efficientnetv2_rw_t）
#         self.student = timm.create_model(
#             "efficientnetv2_rw_t",  # 使用 EfficientNet v2 作為 backbone
#             pretrained=False,  # 不使用預訓練權重
#             in_chans=args.in_channels,  # 輸入通道數，與下游模型一致（如 CT 圖像為 1）
#             drop_rate=0.1,  # Dropout 機率
#             drop_path_rate=0.0  # DropPath 機率
        
#         )
        
#         # 定義教師模型的 backbone，與學生模型一致
#         self.teacher = timm.create_model(
#             "efficientnetv2_rw_t",
#             pretrained=False,
#             in_chans=args.in_channels,
#             drop_rate=0.1,
#             drop_path_rate=0.0
#         )
#         print(f"Student Backbone Output Features: {self.student.num_features}")
#         # 投影頭（Projection Head），將 backbone 特徵映射到對比學習的向量空間
#         self.student_proj_head = ProjectionHead(in_dim=self.student.num_features)  # 使用 EfficientNet 的特徵維度
#         self.teacher_proj_head = ProjectionHead(in_dim=self.teacher.num_features)

#     @torch.no_grad()
#     def _EMA_update_encoder_teacher(self):
#         """
#         使用指數移動平均（EMA）更新教師模型的參數。
#         """
#         momentum = 0.9
#         for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
#             param_t.data = momentum * param_t.data + (1. - momentum) * param.data

#     def forward(self, img, crops, labels):
#         """
#         前向傳播方法。

#         Args:
#             img (torch.Tensor): 隨機裁剪影像，形狀為 [num_slices, num_random_crops, C, H, W]。
#             crops (torch.Tensor): 基礎裁剪影像，形狀為 [num_slices, num_base_crops, C, H, W]。
#             labels (torch.Tensor): 標籤，形狀為 [num_slices, num_random_crops, num_base_crops]。

#         Returns:
#             tuple: 包含正損失、負損失和基礎裁剪正則化損失的元組。
#         """
#         # print(f"Input img shape: {img.shape}")
#         # print(f"Input crops shape: {crops.shape}")
#         # print(f"Input labels shape: {labels.shape}")
#         num_slices, num_random_crops, _, _, _ = img.size()
#         _, num_base_crops, _, _, _ = crops.size()

#         img, crops = img.as_tensor(), crops.as_tensor()
#         # 將 num_slices 和 num_random_crops 或 num_base_crops 合併為批量大小
#         img = img.view(-1, *img.shape[2:])  # [num_slices * num_random_crops, C, H, W]
#         crops = crops.view(-1, *crops.shape[2:])  # [num_slices * num_base_crops, C, H, W]
#         # print(f"Reshaped img shape: {img.shape}")
#         # print(f"Reshaped crops shape: {crops.shape}")
#         # 合併影像輸入，並通過學生模型
#         inputs = torch.cat([img, crops], dim=0)  # [num_slices * (num_random_crops + num_base_crops), C, H, W]
#         student_features = self.student(inputs)
#         print(f"student_features shape ={student_features.shape}")
#         # 更新教師模型參數並獲取特徵
#         self._EMA_update_encoder_teacher()
#         with torch.no_grad():
#             teacher_features = self.teacher(inputs).detach()
#         # print(f"Student features shape: {student_features.shape}")
#         # print(f"Teacher features shape: {teacher_features.shape}")
#         # print(f"len(img) = {len(img)}")
#         # 分解學生和教師特徵
#         # student_img_features = student_features[:len(img)]  # 隨機裁剪的特徵
#         # student_crop_features = student_features[len(img):]  # 基礎裁剪的特徵
#         # teacher_img_features = teacher_features[:len(img)]
#         # teacher_crop_features = teacher_features[len(img):]
#         # 使用ProjectHead
#         student_img_features = self.student_proj_head(student_features[:len(img)])
#         student_crop_features = self.student_proj_head(student_features[len(img):])
#         teacher_img_features = self.teacher_proj_head(teacher_features[:len(img)])
#         teacher_crop_features = self.teacher_proj_head(teacher_features[len(img):])
#         # 初始化損失
#         pos_loss, neg_loss, reg_loss = 0.0, 0.0, 0.0

#         # 按切片進行對比學習
#         for slice_idx in range(num_slices):
#             # 提取當前切片的隨機裁剪和基礎裁剪特徵
#             slice_img_features = student_img_features[slice_idx * num_random_crops:(slice_idx + 1) * num_random_crops]
#             slice_crop_features = student_crop_features[slice_idx * num_base_crops:(slice_idx + 1) * num_base_crops]
#             teacher_slice_img_features = teacher_img_features[slice_idx * num_random_crops:(slice_idx + 1) * num_random_crops]
#             teacher_slice_crop_features = teacher_crop_features[slice_idx * num_base_crops:(slice_idx + 1) * num_base_crops]
#             # print(f"Slice {slice_idx}:")
#             # print(f"  Student img features shape: {slice_img_features.shape}")
#             # print(f"  Student crop features shape: {slice_crop_features.shape}")
#             # print(f"  Teacher img features shape: {teacher_slice_img_features.shape}")
#             # print(f"  Teacher crop features shape: {teacher_slice_crop_features.shape}")

#             # 計算對比學習損失
#             logits1 = self._contrastive_logits(slice_img_features, teacher_slice_crop_features)
#             logits2 = self._contrastive_logits(teacher_slice_img_features,slice_crop_features,)
#             logits = (logits1 + logits2) * 0.5

#             # if slice_idx == 0:
#             #     print('labels and logits:', labels[0].data, logits[0].data)
#             # print(f"  Logits shape: {logits.shape}")
#             # print(f"  Labels shape: {labels[slice_idx].shape}")

#             # 取出對應的標籤
#             slice_labels = labels[slice_idx]  # [num_random_crops, num_base_crops]
#             # print(f"Slice {slice_idx}:")
#             # print(f"  Pos logits (logits1): {logits1}")
#             # print(f"  Neg logits (logits2): {logits2}")
#             # print(f"  Combined logits: {logits}")
#             # print(f"  Labels: {slice_labels}")
#             # 計算損失
#             slice_pos_loss, slice_neg_loss = self._contrastive_loss(slice_labels, logits)
#             pos_loss += slice_pos_loss
#             neg_loss += slice_neg_loss

#             # 基礎裁剪特徵的正則化損失
#             slice_reg_loss = self._regularization_loss(slice_crop_features)
#             reg_loss += slice_reg_loss
#             # breakpoint()
#         # 平均損失
#         pos_loss /= num_slices
#         neg_loss /= num_slices
#         reg_loss /= num_slices
#         # print(f"Pos loss: {pos_loss}, Neg loss: {neg_loss}, Reg loss: {reg_loss}")
#         # breakpoint()
#         return pos_loss, neg_loss, reg_loss

#     def _contrastive_logits(self, img_features, crop_features):
#         """
#         計算隨機裁剪與基礎裁剪特徵之間的相似度。

#         Args:
#             img_features (torch.Tensor): 隨機裁剪特徵，形狀為 [num_random_crops, feature_dim]。
#             crop_features (torch.Tensor): 基礎裁剪特徵，形狀為 [num_base_crops, feature_dim]。

#         Returns:
#             torch.Tensor: 相似度 logits，形狀為 [num_random_crops, num_base_crops]。
#         """
#         logits = F.cosine_similarity(img_features.unsqueeze(1), crop_features.unsqueeze(0), dim=-1)
#         return F.relu(logits)

#     def _contrastive_loss(self, labels, logits):
#         """
#         計算對比學習損失。

#         Args:
#             labels (torch.Tensor): 標籤，形狀為 [num_random_crops, num_base_crops]。
#             logits (torch.Tensor): 預測 logits，形狀為 [num_random_crops, num_base_crops]。

#         Returns:
#             tuple: 包含正損失和負損失的元組。
#         """
#         pos_dis = torch.abs(labels - logits)
#         pos_loss = -labels * torch.log(1 - pos_dis + 1e-6)
#         pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

#         neg_mask = (labels == 0).float()
#         neg_loss = neg_mask * (logits ** 2)
#         neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-6)

#         return pos_loss, neg_loss

#     def _regularization_loss(self, crop_features):
#         """
#         計算基礎裁剪特徵的正則化損失。

#         Args:
#             crop_features (torch.Tensor): 基礎裁剪特徵，形狀為 [num_base_crops, feature_dim]。

#         Returns:
#             torch.Tensor: 正則化損失。
#         """
#         k, _ = crop_features.size()
#         loss = 0.0
#         for i in range(k - 1):
#             for j in range(i + 1, k):
#                 similarity = F.cosine_similarity(crop_features[i], crop_features[j], dim=0)
#                 similarity = F.relu(similarity)  # 使用 ReLU 避免負相似度
#                 loss += similarity ** 2
#         return loss / (k * (k - 1) / 2)


def online_assign(feats, bases):
    b, c = feats.size()
    k, _ = bases.size()
    assert bases.size()[1] == c, print(feats.size(), bases.size())

    logits = []
    for i in range(b):
        feat = feats[i].unsqueeze(0)
        simi = F.cosine_similarity(feat, bases, dim=1).unsqueeze(0)
        logits.append(simi)
    logits = torch.concatenate(logits, dim=0)
    logits = F.relu(logits)

    return logits


def regularization_loss(bases):
    k, c = bases.size()
    loss_all = 0
    num = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            num += 1
            simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
            simi = F.relu(simi)
            loss_all += simi ** 2
    loss_all = loss_all / num

    return loss_all


def ce_loss(labels, logits):
    pos_dis = torch.abs(labels - logits)
    pos_loss = - labels * torch.log(1 - pos_dis + 1e-6)
    pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

    neg_lab = (labels == 0).long()
    neg_loss = neg_lab * (logits ** 2)
    neg_loss = neg_loss.sum() / (neg_lab.sum() + 1e-6)
    return pos_loss, neg_loss



# class VoCoHead(nn.Module):
#     def __init__(self, args):
#         super(VoCoHead, self).__init__()
#         self.student = Swin(args)
#         self.teacher = Swin(args)

#     @torch.no_grad()
#     def _EMA_update_encoder_teacher(self):
#         ## no scheduler here
#         momentum = 0.9
#         for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
#             param_t.data = momentum * param_t.data + (1. - momentum) * param.data

#     def forward(self, img, crops, labels):
#         batch_size = labels.size()[0]
#         total_size = img.size()[0]
#         sw_size = total_size // batch_size
#         pos, neg, total_b_loss = 0.0, 0.0, 0.0

#         img, crops = img.as_tensor(), crops.as_tensor()
#         inputs = torch.cat([img, crops], dim=0)

#         # here we do norm on all instances
#         students_all = self.student(inputs)
#         self._EMA_update_encoder_teacher()
#         with torch.no_grad():
#             teachers_all = (self.teacher(inputs)).detach()

#         x_stu_all, bases_stu_all = students_all[:total_size], students_all[total_size:]
#         x_tea_all, bases_tea_all = teachers_all[:total_size], teachers_all[total_size:]

#         for i in range(batch_size):
#             label = labels[i]
#             # print(label.shape)
#             x_stu, bases_stu = x_stu_all[i * sw_size:(i + 1) * sw_size], bases_stu_all[i * 16:(i + 1) * 16]
#             x_tea, bases_tea = x_tea_all[i * sw_size:(i + 1) * sw_size], bases_tea_all[i * 16:(i + 1) * 16]

#             logits1 = online_assign(x_stu, bases_tea)
#             logits2 = online_assign(x_tea, bases_stu)

#             logits = (logits1 + logits2) * 0.5

#             if i == 0:
#                 print('labels and logits:', label[0].data, logits[0].data)

#             pos_loss, neg_loss = ce_loss(label, logits)
#             pos += pos_loss
#             neg += neg_loss

#             b_loss = regularization_loss(bases_stu)
#             total_b_loss += b_loss

#         pos, neg = pos / batch_size, neg / batch_size
#         total_b_loss = total_b_loss / batch_size

#         return pos, neg, total_b_loss
