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



class ProjectionHead(nn.Module):
    """
    投影頭將 backbone 提取的特徵進一步映射到特定維度（默認 2048），以進行教師-學生模型的對比學習。
    """
    print("ProjectionHead")
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

## VoCo V2 (CNN : b0 v2t v2s)
class VoCoHeadV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_name = "efficientnet_b0"
        self.student = timm.create_model(
            self.model_name,#efficientnetv2_rw_t efficientnet_b0
            pretrained=True,
            in_chans=args.in_channels,
            drop_rate=0.1,
            drop_path_rate=0.0
        )
        self.teacher = timm.create_model(
            self.model_name,#efficientnetv2_rw_t
            pretrained=True,
            in_chans=args.in_channels,
            drop_rate=0.1,
            drop_path_rate=0.0
        )
        print(f"VoCoHeadV2 Student Backbone: {self.model_name}")
        self.student_proj_head = ProjectionHead(in_dim=5120)## b0 5120, v2t 4096 ,v2s 7168
        self.teacher_proj_head = ProjectionHead(in_dim=5120)## b0 5120, v2t 4096 ,v2s 7168

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data
            
    def forward(self, imgs, crops, labels):
        """
        imgs:   [B, 32, 2, 1, 64, 64]
        crops:  [B, 32, 9, 1, 64, 64]
        labels: [B, 32, 2, 9]
        """
        B, S, N_rand, _, H, W = imgs.shape
        _, _, N_base, _, _, _ = crops.shape

        # print(f"🔹 Input imgs shape: {imgs.shape}")
        # print(f"🔹 Input crops shape: {crops.shape}")
        # print(f"🔹 Input labels shape: {labels.shape}")

        # img, crops = img.as_tensor(), crops.as_tensor()

        imgs = imgs.reshape(B * S * N_rand, 1, H, W)
        crops = crops.reshape(B * S * N_base, 1, H, W)
        inputs = torch.cat([imgs, crops], dim=0)

        # print(f"🔸 Combined input shape for CNN: {inputs.shape}")

        student_features = self.student.forward_features(inputs).as_tensor()
        student_features = F.dropout(student_features, p=0.2, training=self.training)
        teacher_features = self.teacher.forward_features(inputs).as_tensor().detach()

        self._EMA_update_encoder_teacher()

        stu_rand = student_features[:B * S * N_rand].view(B, S, N_rand, -1)
        stu_base = student_features[B * S * N_rand:].view(B, S, N_base, -1)
        tea_rand = teacher_features[:B * S * N_rand].view(B, S, N_rand, -1)
        tea_base = teacher_features[B * S * N_rand:].view(B, S, N_base, -1)

        # print(f"🔹 stu_rand shape: {stu_rand.shape}")
        # print(f"🔹 stu_base shape: {stu_base.shape}")
        # print(f"🔹 tea_rand shape: {tea_rand.shape}")
        # print(f"🔹 tea_base shape: {tea_base.shape}")

        # Clip-level mean pooling over 32 slices
        stu_clip = torch.cat([stu_rand, stu_base], dim=2).mean(dim=1)  # [B, N_rand + N_base, dim]
        tea_clip = torch.cat([tea_rand, tea_base], dim=2).mean(dim=1)

        # print(f"🟢 stu_clip shape: {stu_clip.shape}")
        # print(f"🟢 tea_clip shape: {tea_clip.shape}")

        stu_proj = self.student_proj_head(stu_clip)  # [B, N_total, 1024]
        tea_clip = tea_clip.detach() #20250625
        tea_proj = self.teacher_proj_head(tea_clip)

        # print(f"🟢 stu_proj shape: {stu_proj.shape}")
        # print(f"🟢 tea_proj shape: {tea_proj.shape}")

        # Intra-volume contrast
        # print(f"stu_proj[:, :N_rand] = {stu_proj[:, :N_rand].shape}")
        # print(f"tea_proj[:, N_rand:] = {tea_proj[:, N_rand:].shape}")
        logits = self._contrastive_logits_bmm(stu_proj[:, :N_rand], tea_proj[:, N_rand:])
        labels_mean = labels.mean(dim=1)  # [B, 2, 9]

        # print(f"🔻 logits shape (intra): {logits.shape}")
        # print(f"🔻 labels_mean shape: {labels_mean.shape}")

        intra_loss = sum([
            self._contrastive_loss(labels_mean[i], logits[i]) for i in range(B)
        ]) / B

        # Inter-volume contrast
        inter_loss = 0.0
        reg_loss = 0.0
        for i in range(B):
            j = (i + 1) % B

            # A 病患的 random crops
            x_tea = tea_proj[i, :N_rand]        # shape [2, D]
            x_stu = stu_proj[i, :N_rand]        # shape [2, D]

            # B 病患的 base crops
            inter_bases_tea = tea_proj[j, N_rand:]  # shape [9, D]
            inter_bases_stu = stu_proj[j, N_rand:]

            # print(f"x_stu = {x_stu.shape}")
            # print(f"x_tea = {x_tea.shape}")
            # print(f"inter_bases_stu = {inter_bases_stu.shape}")
            # print(f"inter_bases_tea = {inter_bases_tea.shape}")
            inter_loss += self.inter_volume(x_stu, x_tea, inter_bases_stu, inter_bases_tea)

            # Base crop regularization (per patient)
            base_feat = stu_proj[i, N_rand:]  # [N_base, D]
            reg_loss += self._regularization_loss_bmm(base_feat)

        inter_loss /= B
        reg_loss /= B
        
        # reg_loss = self._regularization_loss_bmm(stu_proj[:, N_rand:].reshape(B * N_base, -1))
        # print(f"🧊 reg_loss: {reg_loss.item():.6f}")。

        # print(f"✅ intra_loss: {intra_loss.item():.6f}, inter_loss: {inter_loss.item():.6f}")
        return intra_loss, inter_loss, reg_loss
    
    def inter_volume(self, x_stu, x_tea, inter_bases_stu, inter_bases_tea):
        """
        x_stu: [N_rand, D] → 病患 A 的 random crops（student）
        x_tea: [N_rand, D] → 病患 A 的 random crops（teacher）
        inter_bases_stu: [N_base, D] → 病患 B 的 base crops（student）
        inter_bases_tea: [N_base, D] → 病患 B 的 base crops（teacher）
        """
        pred1 = self._contrastive_logits_bmm(x_tea.unsqueeze(0), inter_bases_tea.unsqueeze(0))  # → [1, N_rand, N_base]
        pred2 = self._contrastive_logits_bmm(x_stu.unsqueeze(0), inter_bases_stu.unsqueeze(0))  # → [1, N_rand, N_base]

        # NaN 偵測與輸出
        if torch.isnan(pred1).any():
            print("❌ NaN detected in pred1 (teacher)")
            print(f"pred1 = {pred1}")
            print(f"x_tea = {x_tea}")
            print(f"inter_bases_tea = {inter_bases_tea}")

        if torch.isnan(pred2).any():
            print("❌ NaN detected in pred2 (student)")
            print(f"pred2 = {pred2}")
            print(f"x_stu = {x_stu}")
            print(f"inter_bases_stu = {inter_bases_stu}")

        inter_loss = self._contrastive_loss(pred1.detach(), pred2)

        if torch.isnan(inter_loss):
            print("❌ NaN detected in inter_loss")
            print(f"pred1 (detached) = {pred1.detach()}")
            print(f"pred2 = {pred2}")

        return inter_loss

    def _contrastive_logits_bmm(self,img_features, crop_features):
        """
        img_features: [B, N1, D]
        crop_features: [B, N2, D]
        Return: [B, N1, N2]
        """

        img_norm = F.normalize(img_features, dim=-1)
        crop_norm = F.normalize(crop_features, dim=-1)
        logits = torch.bmm(img_norm, crop_norm.transpose(1, 2))  # [B, N1, N2]
        return logits
    

    def _contrastive_loss(self, labels, logits):
        #NaN
        logits = logits.clamp(0.0, 1.0)
        labels = labels.clamp(0.0, 1.0)

        pos_dis = torch.abs(labels - logits)
        pos_loss = -labels * torch.log(1 - pos_dis + 1e-6)
        pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

        neg_mask = (labels == 0).float()
        neg_loss = neg_mask * (logits ** 2)
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-6)

        return pos_loss + neg_loss
    
    
    def _regularization_loss_bmm(self, crop_features):
        """
        向量化版本：計算 base crop 特徵間的分散性（避免 collapse）
        Input: crop_features: [K, D]
        Return: scalar loss
        """
        crop_norm = F.normalize(crop_features, dim=1)          # [K, D]
        anchor = crop_norm.detach()                            # 停止其中一邊的梯度
        sim_matrix = torch.matmul(crop_norm, anchor.T)         # [K, K]
        sim_matrix = F.relu(sim_matrix)

        K = crop_features.size(0)
        mask = torch.triu(torch.ones(K, K, device=crop_features.device), diagonal=1)
        selected = sim_matrix[mask.bool()]

        loss = (selected ** 2).mean()
        return loss

## VoCo V1 (CNN : b0 v2t v2s)
class VoCoHeadBatched(nn.Module):
    def __init__(self, args):
        super(VoCoHeadBatched, self).__init__()
        self.model_name = "efficientnet_b0"
        # 設定學生與教師模型（EfficientNetV2）
        self.student = timm.create_model(
            self.model_name, 
            pretrained=True,  
            in_chans=args.in_channels,
            drop_rate=0.1,
            drop_path_rate=0.0
        )
        
        self.teacher = timm.create_model(
            self.model_name,
            pretrained=True,
            in_chans=args.in_channels,
            drop_rate=0.1,
            drop_path_rate=0.0
        )
        print(f"Student Backbone: {self.model_name}")
        print(f"Student Backbone Output Features: {self.student.num_features}")
        # print("Student Conv1 Weights (First 5 Values): ", self.student.state_dict()["conv_stem.weight"].view(-1)[:5])
        # print("Teacher Conv1 Weights (First 5 Values): ", self.teacher.state_dict()["conv_stem.weight"].view(-1)[:5])

        self.student_proj_head = ProjectionHead(in_dim=5120)## b0 5120, v2t 4096 ,v2s 7168
        self.teacher_proj_head = ProjectionHead(in_dim=5120)## b0 5120, v2t 4096 ,v2s 7168

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self):
        momentum = 0.9
        for param, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def forward(self, imgs, crops, labels):
        """
        前向傳播方法

        Args:
            img (torch.Tensor): 隨機裁剪影像，形狀為 [num_slices=32, num_random_crops=2, C=1, H=64, W=64]
            crops (torch.Tensor): 基礎裁剪影像，形狀為 [num_slices=32, num_base_crops=16, C=1, H=64, W=64]
            labels (torch.Tensor): 標籤，形狀為 [num_slices=32, num_random_crops=2, num_base_crops=16]

        Returns:
            tuple: (pos_loss, neg_loss, reg_loss)
        """
        # 保證 batch 維度存在
        if imgs.dim() == 5:
            imgs = imgs.unsqueeze(0)
            crops = crops.unsqueeze(0)
            labels = labels.unsqueeze(0)

        B, S, N_rand, _, H, W = imgs.shape
        _, _, N_base, _, _, _ = crops.shape

        # print(f"🔹 Input img shape: {imgs.shape}")
        # print(f"🔹 Input crops shape: {crops.shape}")
        # print(f"🔹 Input labels shape: {labels.shape}")

        imgs = imgs.reshape(B * S * N_rand, 1, H, W)
        crops = crops.reshape(B * S * N_base, 1, H, W)
        inputs = torch.cat([imgs, crops], dim=0)

        # print(f"🔹 Input imgs shape: {imgs.shape}")
        # print(f"🔹 Input crops shape: {crops.shape}")
        # print(f"🔹 Input labels shape: {labels.shape}")

        imgs = imgs.reshape(B * S * N_rand, 1, H, W)
        crops = crops.reshape(B * S * N_base, 1, H, W)
        inputs = torch.cat([imgs, crops], dim=0)

        # print(f"🔸 Combined input shape for CNN: {inputs.shape}")

        student_features = self.student.forward_features(inputs).as_tensor()
        self._EMA_update_encoder_teacher()
        with torch.no_grad():
            teacher_features = self.teacher.forward_features(inputs).as_tensor().detach()

        stu_rand = student_features[:B * S * N_rand].view(B, S, N_rand, -1)
        stu_base = student_features[B * S * N_rand:].view(B, S, N_base, -1)
        tea_rand = teacher_features[:B * S * N_rand].view(B, S, N_rand, -1)
        tea_base = teacher_features[B * S * N_rand:].view(B, S, N_base, -1)

        # print(f"🔹 stu_rand shape: {stu_rand.shape}")
        # print(f"🔹 stu_base shape: {stu_base.shape}")
        # print(f"🔹 tea_rand shape: {tea_rand.shape}")
        # print(f"🔹 tea_base shape: {tea_base.shape}")

        # Clip-level mean pooling over 32 slices
        stu_clip = torch.cat([stu_rand, stu_base], dim=2).mean(dim=1)  # [B, N_rand + N_base, dim]
        tea_clip = torch.cat([tea_rand, tea_base], dim=2).mean(dim=1)

        # print(f"🟢 stu_clip shape: {stu_clip.shape}")
        # print(f"🟢 tea_clip shape: {tea_clip.shape}")

        stu_proj = self.student_proj_head(stu_clip)  # [B, N_total, 1024]
        tea_clip = tea_clip.detach() ### 20250625
        tea_proj = self.teacher_proj_head(tea_clip)

        # print(f"🟢 stu_proj shape: {stu_proj.shape}")
        # print(f"🟢 tea_proj shape: {tea_proj.shape}")

        
        # print(f"🟢 stu_proj[:, :N_rand] shape: {stu_proj[:, :N_rand].shape}")
        # print(f"🟢 tea_proj[:, N_rand:] shape: {tea_proj[:, N_rand:].shape}")  

        # **確保 logits 計算正確**
        logits1 = self._contrastive_logits(stu_proj[:, :N_rand], tea_proj[:, N_rand:])
        logits2 = self._contrastive_logits(tea_proj[:, :N_rand], stu_proj[:, N_rand:] )
        # logits1 = self._contrastive_logits(student_clip_features[:num_random_crops], teacher_clip_features[num_random_crops:])
        # logits2 = self._contrastive_logits(teacher_clip_features[:num_random_crops], student_clip_features[num_random_crops:])
        logits = (logits1 + logits2) * 0.5
        # print(f"🔴 logits shape: {logits.shape}")

        # **確保 labels 形狀匹配**
        labels_mean = labels.mean(dim=1)
        # print(f"🟠 labels shape after mean: {labels_mean.shape}")

        # **檢查 labels 是否與 logits 對應**
        # assert logits.shape == labels_mean.shape, f"Mismatch! logits: {logits.shape}, labels: {labels_mean.shape}"

        # intra_loss = sum([
        #     self._contrastive_loss(labels_mean[i], logits[i]) for i in range(B)
        # ]) / B
        #---#
        total_pos_loss = 0.0
        total_neg_loss = 0.0

        for i in range(B):
            pos_loss, neg_loss = self._contrastive_loss(labels_mean[i], logits[i])
            total_pos_loss += pos_loss
            total_neg_loss += neg_loss

        # 平均
        pos_loss = total_pos_loss / B
        neg_loss = total_neg_loss / B
        # print(f"after average = {pos_loss.item(),neg_loss.item()}")
        #---#
        
        reg_loss = 0.0
        for i in range(B):
            base_feat = stu_proj[i, N_rand:]  # [N_base, D]
            reg_loss += self._regularization_loss(base_feat)
        reg_loss /= B
        # **保留 `reg_loss`**
        # reg_loss = self._regularization_loss(student_clip_features[num_random_crops:])
        inter_loss = 0.0
        inter_loss = torch.tensor(inter_loss)
        # breakpoint()  # ⬅ 設定 Debug 停止點
        return pos_loss, neg_loss, reg_loss
        # return intra_loss, inter_loss, reg_loss
    
    def _contrastive_logits(self, img_features, crop_features):
        """
        確保 logits 形狀正確
        """
        # print(f"🔻 Contrastive logits input shapes: img_features: {img_features.shape}, crop_features: {crop_features.shape}")
        img_norm = F.normalize(img_features, dim=-1)
        crop_norm = F.normalize(crop_features, dim=-1)
        logits = torch.bmm(img_norm, crop_norm.transpose(1, 2))  # [B, N1, N2]
        # print(f"🔺 logits output shape: {logits.shape}")
        return logits

    def _contrastive_loss(self, labels, logits):
        """
        計算對比學習損失。
        """
        # print(f"🟣 Contrastive loss input shapes: labels: {labels.shape}, logits: {logits.shape}")
        
        pos_dis = torch.abs(labels - logits)
        pos_loss = -labels * torch.log(1 - pos_dis + 1e-6)
        pos_loss = pos_loss.sum() / (labels.sum() + 1e-6)

        neg_mask = (labels == 0).float()
        neg_loss = neg_mask * (logits ** 2)
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-6)

        # return pos_loss + neg_loss
        # print(pos_loss.item(), neg_loss.item())
        return pos_loss, neg_loss
        
    def _regularization_loss(self, crop_features):#20250625
        crop_norm = F.normalize(crop_features, dim=1)         # [K, D]
        crop_anchor = crop_norm.detach()                      # 停止其中一邊的梯度

        sim_matrix = torch.matmul(crop_norm, crop_anchor.T)   # [K, K]，只有左邊會接收梯度
        sim_matrix = F.relu(sim_matrix)

        K = crop_features.size(0)
        mask = torch.triu(torch.ones(K, K, device=crop_features.device), diagonal=1)
        selected = sim_matrix[mask.bool()]

        loss = (selected ** 2).mean()
        return loss

