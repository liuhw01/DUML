from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torch
from torch.nn import functional as F
from attention import CBAM
from attention import CBAM_O
from attention import OcclusionGate
import numpy as np
import torchvision.ops as ops
import torchsnooper
from functions import ReverseLayerF
from torch.nn import Parameter
import math



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class AttentionBlock(nn.Module):
    __constants__ = ['downsample']
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out



class ADDNET(nn.Module):
    def __init__(self,AttentionBlock):
        super(ADDNET, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Shared feature extraction module.
        self.layer3 = self._make_layer(AttentionBlock,128, 256, 6,stride=2)  # 14x14x256
        self.layer4 = self._make_layer(AttentionBlock,256, 512, 3, stride=2)  # 第一个stride=2,剩下3个stride=1;7x7x512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Global.
        out_g = self.layer3(x)  # 14x14x256
        out_g = self.layer4(out_g)  #7x7x512
        out_g = self.avgpool(out_g)
        out_g = torch.flatten(out_g, 1)
        return out_g


class GlobalNet(nn.Module):
    def __init__(self,BasicBlock):
        super(GlobalNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Shared feature extraction module.
        self.layer5_1 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 第一个stride=2,剩下3个stride=1;7x7x128
        self.layer5_2 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 第一个stride=2,剩下3个stride=1;7x7x128
        self.layer5_3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 第一个stride=2,剩下3个stride=1;7x7x128
        self.layer5_4 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)  # 第一个stride=2,剩下3个stride=1;7x7x128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        # Local.
        patch_11 = x[:, :, 0:14, 0:14]
        patch_12 = x[:, :, 0:14, 14:28]
        patch_21 = x[:, :, 14:28, 0:14]
        patch_22 = x[:, :, 14:28, 14:28]
        out_l11 = self.layer5_1(patch_11)
        out_l12 = self.layer5_2(patch_12)
        out_l21 = self.layer5_1(patch_21)
        out_l22 = self.layer5_2(patch_22)
        out_l11 = self.avgpool(out_l11)
        out_l11 = torch.flatten(out_l11, 1)
        out_l12 = self.avgpool(out_l12)
        out_l12 = torch.flatten(out_l12, 1)
        out_l21 = self.avgpool(out_l21)
        out_l21 = torch.flatten(out_l21, 1)
        out_l22 = self.avgpool(out_l22)
        out_l22 = torch.flatten(out_l22, 1)
        out= torch.cat([out_l11,out_l12,out_l21,out_l22],dim=1)
        return out




class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50, easy_margin=False, use_gpu=True):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # https://www.jianshu.com/p/d8b77cc02410
        if self.use_gpu:
            self.weight = nn.Parameter(torch.randn(num_classes, in_features).cuda())
        else:
            self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        log_probs = self.logsoftmax(output)
        return F.nll_loss(log_probs,label)




class MSFER(nn.Module):
    def __init__(self,  num_classes=7, num_domains = 3,nfeat=512):
        super(MSFER, self).__init__()
        self.num_domains = num_domains
        '''Net of MSFER'''
        norm_layer = nn.BatchNorm2d
        self.num_classes=num_classes
        self.beta=0.5
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.nfeat=nfeat
        self.batch=np.ones(self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        # Source domain feature prototype.
        self.mean_sour = list()
        for i in range(self.num_domains):
            self.mean_sour.append(torch.zeros(self.num_classes, self.nfeat).cuda())
        # Target domain feature prototype.
        self.mean_tar = list()
        for i in range(self.num_domains):
            self.mean_tar.append(torch.zeros(self.num_classes, self.nfeat).cuda())
        # Common features.
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 3)  # 56x56x64
        self.layer2 = self._make_layer(BasicBlock, 64, 128, 4, stride=2)  #

        self.domain_classifier = nn.Linear(128, self.num_domains)
        # Specific features.

        #self.addnetlist = ADDNET(AttentionBlock)
        self.addnetlist= nn.ModuleList([ADDNET(AttentionBlock) for i in range(self.num_domains)])
        # Specific classifier.
        #self.classifier =nn.Linear(self.nfeat, self.num_classes)
        self.classifier= nn.ModuleList([nn.Linear(self.nfeat, self.num_classes) for i in range(self.num_domains)])
        # Domain discriminator.
        self.domain_fcnetlist = nn.ModuleList([nn.Linear(self.nfeat, 2) for i in range(self.num_domains)])
        # Domain-shared fine-grained features.
        self.globalNet=GlobalNet(AttentionBlock)
        self.global_classifer=nn.Linear(1024,self.num_classes)

        self.Arcloss=ArcFaceLoss(self.nfeat,self.num_classes )
        self.mse_loss = nn.MSELoss()
        self.KL_loss=nn.KLDivLoss()
        self.CE=nn.CrossEntropyLoss()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    # update prototypes and local relation alignment loss
    def update_statistics(self, feats, labels,domain_idx, inn,epsilon=1e-5):
        num_labels = 0
        loss_local = 0
        tmp_feat = feats
        tmp_label=labels
        #tmp_label = torch.tensor(labels,dtype = torch.int64)
        num_labels += tmp_label.shape[0]
        onehot_label = torch.zeros((tmp_label.shape[0], int(self.num_classes))).scatter_(1,tmp_label.unsqueeze(-1).cpu(),1).float().cuda()
        domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
        tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)
        tmp_mask = (tmp_mean.sum(-1) != 0).float().unsqueeze(-1)
        if inn=='sour':
            self.mean_sour[domain_idx] = self.mean_sour[domain_idx].detach() * (1 - tmp_mask) + (
                    self.mean_sour[domain_idx].detach() * self.beta + tmp_mean * (
                        1 - self.beta)) * tmp_mask
            domain_feature_center = onehot_label.unsqueeze(-1) * self.mean_sour[domain_idx].unsqueeze(0)
            tmp_mean_center = domain_feature_center.sum(1)
            # compute local relation alignment loss
            loss_local += (((tmp_mean_center - tmp_feat) ** 2).mean(-1)).sum()
        if inn=='tar':
            self.mean_tar[domain_idx] = self.mean_tar[domain_idx].detach() * (1 - tmp_mask) + (self.mean_tar[domain_idx].detach() * self.beta + tmp_mean * (1 - self.beta)) * tmp_mask
            domain_feature_center = onehot_label.unsqueeze(-1) * self.mean_tar[domain_idx].unsqueeze(0)
            tmp_mean_center = domain_feature_center.sum(1)
            # compute local relation alignment loss
            loss_local += (((tmp_mean_center - tmp_feat) ** 2).mean(-1)).sum()
        return self.mean_sour, self.mean_tar, loss_local/num_labels

    def entropy(self, Plist):
        entropy_resut=[]
        for x in Plist:
            result = 0
            for x1 in x:
                if x1<=0:
                    x1=torch.tensor(1e-6).cuda()
                result += (-x1) * math.log2(x1)
            entropy_resut.append(torch.detach(result))
        return entropy_resut
    def forward(self,data_src, data_tgt=0,  label_src=0, mark=0,epoch=1,alpha=1,confident=0.4,Temp=4,nask=0.3010, Coefficient=500):
        ######  Common features. ##########
        # Common features. data_src和data_tgt
        # source
        data_src = self.conv1(data_src)
        data_src = self.bn1(data_src)
        data_src = self.relu(data_src)
        data_src = self.maxpool(data_src)
        # 56x56x64
        data_src = self.layer1(data_src)  # 56x56x64
        data_src = self.layer2(data_src)  # 28x28x128


        #Target domain features.
        tar_fea_domain=[]
        #Target domain prediction.
        tar_pre_domain=[]
        tar_pre_domain_cla=[]
        #Target domain
        tar_dom_domian=[]
        tar_dom_pre_cla=[]

        for i in range(self.num_domains):
            tar_fea=self.addnetlist[i](data_src)
            tar_pre = self.classifier[i](tar_fea)
            tar_fea = tar_fea.view(tar_fea.size(0), -1)
            reverse_feature = ReverseLayerF.apply(tar_fea, alpha)
            tar_dom= self.domain_fcnetlist[i](reverse_feature)
            tar_dom_pre=F.softmax(tar_dom)
            tar_fea_domain.append(tar_fea)
            tar_pre_domain.append(tar_pre)
            tar_pre_domain_cla.append(F.softmax(tar_pre))
            tar_dom_domian.append(tar_dom)
            tar_dom_pre_cla.append(tar_dom_pre)

        # Target domain pseudo-labels
        # Domain similarity
        # Prediction confidence
        # Domain confidence

        sour_dom_cla = self.avgpool(data_src)
        sour_dom_cla = torch.flatten(sour_dom_cla, 1)
        sour_dom_cla = self.domain_classifier(sour_dom_cla)
        D_confidence = [2 / (1 + math.exp(-10 * (self.batch[i]) / (max(self.batch)))) - 1 for i in
                     range(self.num_domains)]

        P_confidence = []
        for i in range(len(tar_dom_pre_cla)):
            weii = sum(self.entropy(tar_dom_pre_cla[i])) / len(tar_dom_pre_cla[i])
            P_confidence.append(math.exp(-(nask - weii) * Coefficient))
        confidence=[P_confidence[i]/sum(P_confidence) for i in range(len(P_confidence))]

        cer_dom_e = []
        for i in range(len(tar_pre_domain_cla)):
            cer = self.entropy(tar_pre_domain_cla[i])
            cer_e = [1/cer[j] for j in range(len(cer))]
            cer_dom_e.append(cer_e)

        D_similarity = torch.transpose(torch.tensor(cer_dom_e), dim0=1, dim1=0)
        for d_sim in range(len(D_similarity)):
            for sub_d_sim in range(len(D_similarity[0])):
                D_similarity[d_sim][sub_d_sim] = D_similarity[d_sim].sum()/D_similarity[d_sim][sub_d_sim]
        D_similarity = D_similarity.sum(dim=0)
        D_similarity = [D_similarity[i] / sum(D_similarity) for i in range(len(D_similarity))]
        # Pseudo label
        Pseudo_label = []
        for i in range(len(tar_pre_domain_cla[0])):
            aplh = [confidence[j] * D_confidence[j] * D_similarity[j] for j in range(len(tar_pre_domain))]
            aplh_sum = sum(aplh)
            Pseudo = torch.stack(
                [aplh[j] / aplh_sum * tar_pre_domain_cla[j][i] for j in range(len(tar_pre_domain))])
            Pseudo_label.append(torch.sum(Pseudo, axis=0))

        sour_glob_fea = self.globalNet(data_src)
        sour_glob_cla = self.global_classifer(sour_glob_fea)



        return sour_glob_cla, tar_pre_domain, torch.stack(Pseudo_label)


