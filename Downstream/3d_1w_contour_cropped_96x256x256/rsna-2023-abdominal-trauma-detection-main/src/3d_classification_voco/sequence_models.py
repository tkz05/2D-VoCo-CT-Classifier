import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from heads import ClassificationHead
from unet import Encoder  # 確保 Encoder 定義正確


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
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class RNNClassificationModel(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, feature_pooling_type, rnn_class, rnn_args, dropout_rate, freeze_parameters):

        super(RNNClassificationModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )
        ### Load backbone
        #----------------------------------------------------#
        print("Load backbone")
        # student_weights_path = '/workspace/rsna/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t/20250722_only_rsna_b0_v2(b4_3x3)_step60000.pth'
        # student_weights_path = '/workspace/rsna/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t/20250721_only_rsna_b0_v1(b1_4x4)_step30000.pth'
        # student_weights_path = '/workspace/rsna/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t/20250717_only_rsna_v2t_v2(b4_3x3)_step60000.pth'
        student_weights_path = '/workspace/rsna/rsna-2023-abdominal-trauma-detection-main/models/voco_lstm_efficientnetv2t/20250718_only_rsna_v2s_v1(b1_4x4)_step30000.pth'

        print(student_weights_path.split("/")[-1])
        pretrained_weights = torch.load(student_weights_path)
        cnn_weights = {k.replace('student.', ''): v for k, v in pretrained_weights.items() if k.startswith('student.')}
        # cnn_weights = {k.replace('student.backbone.', ''): v for k, v in pretrained_weights.items() if k.startswith('student.backbone.')} #v4
        missing_keys, unexpected_keys = self.backbone.load_state_dict(cnn_weights, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        # ----------------------------------------------------#

        # freeze_parameters = True
        if freeze_parameters:
            print("Freeze part of the backbone!")
            unfreeze_layers = 3
            self.freeze_backbone_layers(unfreeze_layers)
        else :
            print("no freeze!")

        self.feature_pooling_type = feature_pooling_type
        input_features = self.backbone.get_classifier().in_features
        self.backbone.classifier = nn.Identity()

        self.pooling = nn.Identity()

        self.rnn = getattr(nn, rnn_class)(input_size=input_features, **rnn_args)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        input_dimensions = rnn_args['hidden_size'] * (int(rnn_args['bidirectional']) + 1)
        self.head = ClassificationHead(input_dimensions=input_dimensions)

    def freeze_backbone_layers(self, unfreeze_layers=3):
        """
        凍結 EfficientNet Backbone，但允許 fine-tune 最後 N 層。
        """
        # 先凍結整個 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 取得 EfficientNet 的 block 列表
        blocks = list(self.backbone.blocks)  
        unfreeze_start = len(blocks) - unfreeze_layers  # 計算要解凍的起點

        # 解凍最後 unfreeze_layers 個 blocks
        for i, block in enumerate(blocks):
            if i >= unfreeze_start:
                print(f"🔓 解凍 Block {i}")
                for param in block.parameters():
                    param.requires_grad = True

        # **確保解凍 conv_head 和 classifier**
        for name, param in self.backbone.named_parameters():
            if "conv_head" in name or "classifier" in name:
                print(f"🔓 解凍 {name}")
                param.requires_grad = True

        # **確保 `bn2.weight` 和 `bn2.bias` 被解凍**
        # for name, param in self.backbone.named_parameters():
        #     if name in ["bn2.weight", "bn2.bias"]:
        #         print(f"🔓 解凍 {name}")
        #         param.requires_grad = True

        # **解凍 classifier 層**
        if hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
            print("🔓 解凍 classifier 層")



    def forward(self, x):

        input_batch_size, input_channel, input_depth, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_depth, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)

        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.feature_pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)
        elif self.feature_pooling_type == 'gem':
            x = self.pooling(x).view(x.size(0), -1)
        else:
            raise ValueError(f'Invalid feature pooling type {self.feature_pooling_type}')

        x = x.contiguous().view(input_batch_size, input_depth, feature_channel)
        x, _ = self.rnn(x)
        x = torch.max(x, dim=1)[0]
        x = self.dropout(x)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output

class CNNClassificationModel(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, feature_pooling_type, dropout_rate, freeze_parameters):

        super(CNNClassificationModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.feature_pooling_type = feature_pooling_type
        input_features = self.backbone.get_classifier().in_features
        self.backbone.classifier = nn.Identity()

        if self.feature_pooling_type == 'gem':
            self.pooling = GeM()
        else:
            self.pooling = nn.Identity()

        self.cnn = CNN(in_channels=96)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features)

    def forward(self, x):

        input_batch_size, input_channel, input_depth, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_depth, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)

        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.feature_pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)
        elif self.feature_pooling_type == 'gem':
            x = self.pooling(x).view(x.size(0), -1)
        else:
            raise ValueError(f'Invalid feature pooling type {self.feature_pooling_type}')

        x = x.contiguous().view(input_batch_size, input_depth, feature_channel)
        x = self.cnn(x).view(input_batch_size, feature_channel)
        x = self.dropout(x)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output
