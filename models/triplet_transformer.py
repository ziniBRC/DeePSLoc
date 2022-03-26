import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.resnetlab import resnet18, ASPP_module, BatchNorm2d
from models.transformer import Transformer


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1, similar=True, threshold=4):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.similar = similar
        self.threshold = threshold

    def forward(self, q, k, v, attn_mask=None, train_attn=False):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        # / self.scale_factor
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor

        # attn: [b_size x len_q x len_k]
        # attn = torch.relu(attn + self.threshold) - self.threshold
        # attn = -torch.relu(-attn + self.threshold) + self.threshold
        if attn_mask is not None:
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # attn = self.dropout(attn)
        n_heads = attn.size(0)
        v = v.repeat(n_heads, 1, 1)
        if train_attn is False:
            attn = attn.detach()
            outputs = torch.bmm(attn, v)
        else:
            v = v.detach()
            outputs = torch.bmm(attn, v)
        # outputs: [b_size x len_q x d_v]

        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, dropout=0.1):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.layer_norm1 = nn.LayerNorm(d_k)
        self.layer_norm2 = nn.LayerNorm(d_k)

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)

    def forward(self, q, k, proj_v, attn_mask=None, train_attn=False):
        (d_k, d_model, n_heads) = (self.d_k, self.d_model, self.n_heads)
        b_size = q.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        # [n_heads x b_size * len_k x d_model]

        q_s = torch.bmm(q_s, self.w_q)
        q_s = self.layer_norm1(q_s)
        q_s = q_s.view(b_size * n_heads, -1, d_k)
        # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k)
        k_s = self.layer_norm2(k_s)
        k_s = k_s.view(b_size * n_heads, -1, d_k)
        # [b_size * n_heads x len_k x d_k]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, proj_v, attn_mask=attn_mask, train_attn=train_attn)

        # return a list of tensors [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class TripletFormer(nn.Module):
    def __init__(self, dim=256, dropout=0.1, num_heads=1, num_layer=1, num_classes=2):
        super(TripletFormer, self).__init__()
        model_dim = dim
        # QUERY_DIM = 32
        key_dim = dim // 4
        value_dim = dim
        FF_DIM = model_dim * 4

        self.attn_layer_ap = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)
        self.attn_layer_an = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)

        self.proj = nn.Linear(model_dim, num_classes)

    def forward(self, embedding_apn):
        # enc_outputs = enc_inputs
        # # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # enc_self_attn_mask = None
        (embedding_a, embedding_p, embedding_n) = embedding_apn

        output_p, ap_attn = self.attn_layer_ap(embedding_a, embedding_p, embedding_p)
        output_n, an_attn = self.attn_layer_an(embedding_a, embedding_p, embedding_p)

        out = self.proj(embedding_a)
        return out, output_p, output_n


# class ResNetTripletFormer(nn.Module):
#     def __init__(self, num_classes=2, feat_dim=256, num_heads=1, dropout=0.25, device=None):
#         super(ResNetTripletFormer, self).__init__()
#         model_dim = feat_dim
#         # 3QUERY_DIM = 32
#         key_dim = 128
#         value_dim = feat_dim
#         FF_DIM = model_dim * 4
#
#         self.feat_extractor = resnetlab(n_classes=num_classes)
#         self.transformer = Transformer(fv='res18-' + str(feat_dim), NUM_CLASSES=num_classes, NUM_HEADS=4)
#         self.device = device
#
#         # self.triplet_former = TripletFormer(num_classes=num_classes)
#         self.attn_layer_ap = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)
#         self.attn_layer_an = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)
#
#         self.proj = nn.Linear(feat_dim, feat_dim)
#         self.classifier = nn.Linear(feat_dim, num_classes)
#
#     def forward(self, imgs, pos_mask=None, training=True, train_attn=False, ):
#         if training is True:
#             _, features, patch_feats = self.feat_extractor(imgs)
#             _, features = self.transformer(patch_feats.view(patch_feats.shape[0], patch_feats.shape[1], -1).transpose(1, 2))
#
#             proj_feats = self.proj(features)
#             proj_feats = proj_feats.unsqueeze(0)
#             features = features.unsqueeze(0)
#             if train_attn is True:
#                 proj_feats = proj_feats.detach()
#                 features = features.detach()
#
#             # ap_attn_mask = (~pos_mask + torch.eye(pos_mask.shape[0], dtype=torch.bool).to(self.device)).unsqueeze(0)
#             ap_attn_mask = (~pos_mask).unsqueeze(0)
#             an_attn_mask = pos_mask.unsqueeze(0)
#             output_p, ap_attn = self.attn_layer_ap(features, features, features, train_attn=train_attn, attn_mask=ap_attn_mask)
#             output_n, an_attn = self.attn_layer_an(features, features, features, train_attn=train_attn, attn_mask=an_attn_mask)
#
#             output_a = self.classifier(features.squeeze(0))
#
#             return output_a, features, output_p, output_n
#         else:
#             _, features, patch_feats = self.feat_extractor(imgs)
#             _, features = self.transformer(patch_feats.view(patch_feats.shape[0], patch_feats.shape[1], -1).transpose(1, 2))
#             # proj_a = self.proj(feature_a)
#             output_a = self.classifier(features)
#             return output_a, features


class ASPProj(nn.Module):
    def __init__(self):
        super(ASPProj, self).__init__()
        dilations = [1, 2, 4]
        self.aspp1 = ASPP_module(512, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(512, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(512, 256, dilation=dilations[2])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())
        self.last_conv = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x5), dim=1)
        feature = self.last_conv(x)
        feature = feature.view(feature.size(0), -1)
        return feature


class ResNetTripletFormer(nn.Module):
    def __init__(self, num_classes=2, feat_dim=256, num_heads=6, dropout=0.25, device=None):
        super(ResNetTripletFormer, self).__init__()
        model_dim = feat_dim
        # 3QUERY_DIM = 32
        key_dim = 128
        value_dim = feat_dim
        FF_DIM = model_dim * 4
        self.n_heads = num_heads
        self.d_model = model_dim
        self.device = device

        self.feat_extractor = resnet18(nc=num_classes, pretrained=True)
        self.proj = ASPProj()

        # self.triplet_former = TripletFormer(num_classes=num_classes)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.attn_layer_ap = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)
        self.attn_layer_an = _MultiHeadAttention(key_dim, model_dim, num_heads, dropout)

        # self.w_v = nn.Parameter(torch.FloatTensor(num_heads, model_dim, model_dim))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, imgs, same_mask=None, training=True, train_attn=False, ):
        if training is True:
            features = self.feat_extractor(imgs)

            proj_feats = self.proj(features)
            if train_attn is True:
                proj_feats = proj_feats.detach()
                features = features.detach()

            features = self.feat_proj(features).view(features.size(0), -1)
            proj_feats = proj_feats.unsqueeze(0)
            features = features.unsqueeze(0)
            ap_attn_mask = (~same_mask).unsqueeze(0)
            an_attn_mask = same_mask.unsqueeze(0)

            output_p, ap_attn = self.attn_layer_ap(features, features, proj_feats, train_attn=train_attn, attn_mask=ap_attn_mask)
            output_n, an_attn = self.attn_layer_an(features, features, proj_feats, train_attn=train_attn, attn_mask=an_attn_mask)

            output_a = self.classifier(proj_feats.squeeze(0))

            return output_a, proj_feats, output_p, output_n
        else:
            feature_a = self.feat_extractor(imgs)
            proj_a = self.proj(feature_a)
            output_a = self.classifier(proj_a)
            return output_a, proj_a
