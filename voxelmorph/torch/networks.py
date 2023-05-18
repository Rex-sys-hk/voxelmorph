import numpy as np
from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from voxelmorph.py.utils import dice

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
import matplotlib.pyplot as plt


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()
        print('Loading VxmDense model--Pytorch version')
        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer 
        # mode have to be bilinear when training
        # mode have to be nearest when testing
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, *args, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        if self.training:
            self.transformer.mode = 'bilinear'
        else:
            self.transformer.mode = 'nearest'
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, target)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class EncoderDecoder(Unet):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        super().__init__(
                 inshape,
                 infeats,
                 nb_features,
                 nb_levels,
                 max_pool,
                 feat_mult,
                 nb_conv_per_level,
                 half_res)

    def encoding(self, x):
        # encoder forward pass
        self.x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            self.x_history.append(x)
            x = self.pooling[level](x)
        return x


    def decoding(self, x):
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, self.x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
        return x

class VxmComp(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        print('Loading VxmComp model--Pytorch version')
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core unet model
        self.unet_model = EncoderDecoder(
            inshape,
            infeats=(src_feats + trg_feats),
            # infeats=src_feats,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.augmentation_mode = 'focus_trans' #'fuzzy_comp', 'seg_wise','focus_trans'
        # self.mlp = nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Dropout(0.1),nn.Linear(32,32))

    def forward(self, source, target, source_label, target_label, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        # concatenate inputs and propagate unet
        btsz = source.shape[0]
        ## SS seperate segmentation
        if self.augmentation_mode=='seg_wise' and self.training and torch.rand(1) > 0.5:
            source1 = source.clone()
            source2 = source.clone()
            target1 = target.clone()
            target2 = target.clone()
            label_1 = torch.randint(1,24,(btsz,1,1,1)).to(source.device)
            s_label = torch.where(source_label == label_1, True, False)
            t_label = torch.where(target_label == label_1, True, False)
            source1[~s_label] = 0
            target1[~t_label] = 0
            label_2 = torch.randint(1,24,(btsz,1,1,1)).to(source.device)
            s_label = torch.where(source_label == label_2, True, False)
            t_label = torch.where(target_label == label_2, True, False)
            source2[~s_label] = 0
            target2[~t_label] = 0
            inputs1 = torch.cat([source1, target1], dim=1)
            inputs2 = torch.cat([source2, target2], dim=1)
            # inputs = torch.cat([source, target], dim=1)
            inputs = torch.cat([inputs1,inputs2],dim=0)
            x = self.unet_model.encoding(inputs)
            x1 = x[:btsz]
            x2 = x[btsz:]
            target = torch.cat([target1,target2],dim=0)
            # if label_1 == label_2:
            #     self.comp_loss = torch.norm(self.mlp(x1.permute(0,2,3,1))**2-self.mlp(x2.permute(0,2,3,1))**2,dim=[-1]).mean()
            # else:
            #     self.comp_loss = -torch.norm(self.mlp(x1.permute(0,2,3,1))**2-self.mlp(x2.permute(0,2,3,1))**2,dim=[-1]).mean().clamp(max=20)
            if label_1 == label_2:
                self.comp_loss = torch.norm(x1-x2,dim=[-1]).mean()
            else:
                self.comp_loss = -torch.norm(x1-x2,dim=[-1]).mean().clamp(max=10)

        ## Fuzzy Contrastive
        elif self.training and btsz==2 and self.augmentation_mode == 'fuzzy_comp':
            inputs = torch.cat([source, target], dim=1)
            x = self.unet_model.encoding(inputs)
            overlap_s = torch.zeros_like(source_label[...,0:1,0].repeat(1,1,24))
            bottoms = torch.zeros_like(source_label[...,0:1,0].repeat(1,1,24))
            # overlap_t = torch.zeros_like(source_label[...,0:1,0].repeat(1,1,24))
            s_size = torch.where(source_label == 0, False, True).sum(dim=[-1,-2])
            t_size = torch.where(target_label == 0, False, True).sum(dim=[-1,-2])

            for idx, label in enumerate(range(1,24)):
                s_label = torch.where(source_label == label, True, False)
                t_label = torch.where(target_label == label, True, False)

                top = 2 * torch.sum(torch.logical_and(s_label, t_label),dim=[-1,-2])
                bottom = torch.sum(s_label,dim=[-1,-2]) + torch.sum(t_label,dim=[-1,-2])
                diff = torch.sum(t_label,dim=[-1,-2])-torch.sum(s_label,dim=[-1,-2])
                # bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
                # dicem[idx] = top / bottom
                bottoms[:,:,idx] = bottom
                overlap_s[:,:,idx] = top/bottom
                overlap_s = overlap_s.nan_to_num(nan=0.0)
                # overlap_t[:,idx] = torch.sum(overlap.astype(float),dim=[-1,-2,-3])/torch.sum(t_label.astype(float),dim=[-1,-2,-3])-0.5
            # overlap_s = overlap_s-overlap_s.mean(dim=-1, keepdim=True)

            # min_bot, min_bot_id = bottoms.topk(k=5,dim=-1,largest=False)
            # overlap_s = overlap_s.gather(dim=-1,index=min_bot_id)
            sim_score = self.cos(overlap_s[0], overlap_s[1])
            self.comp_loss = (sim_score*torch.norm(x[0]-x[1],dim=[-3])).mean().clamp(min=-10)
        ## FT
        elif self.training and btsz==2 and self.augmentation_mode == 'focus_trans':
            inputs = torch.cat([source, target], dim=1)
            x = self.unet_model.encoding(inputs)
            x_orig = x.clone()
            masked_lable_id = torch.randint(1, 24, source[:,:,0:1,0:1].shape, device=source.device)
            if torch.rand([1])<0.5:
                seg_change = True if self.training else False
            else:
                seg_change = False
            if torch.rand([1])<0.5 and seg_change:
                source[source_label==masked_lable_id] = 0.
            elif seg_change:
                target[target_label==masked_lable_id] = 0.
            inputs = torch.cat([source, target], dim=1)
            x_diff = self.unet_model.encoding(inputs)
            x_orig_n = torch.nn.functional.normalize(x_orig,dim=[-3])
            x_diff_n = torch.nn.functional.normalize(x_diff,dim=[-3])
            sim_loss = (torch.norm(x_orig_n-x_diff_n,dim=[-3])**2).clamp(min=0).mean()
            diff_loss = (1-torch.norm(x_orig_n-x_diff_n,dim=[-3])**2).clamp(min=0).mean()
            self.comp_loss = sim_loss if seg_change else diff_loss
        else:
            inputs = torch.cat([source, target], dim=1)
            x = self.unet_model.encoding(inputs)
            self.comp_loss = torch.zeros([1]).to(source.device).mean(0)

        x = self.unet_model.decoding(x)
        # self.l_x = x_orig_n.detach()
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        if self.training:
            self.transformer.mode = 'bilinear'
        else:
            self.transformer.mode = 'nearest'
        y_source = self.transformer(inputs[:,:btsz], pos_flow)
        y_target = self.transformer(inputs[btsz:,1:], neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow, target)
        else:
            return y_source, pos_flow

    def get_comp_loss(self):
        return self.comp_loss