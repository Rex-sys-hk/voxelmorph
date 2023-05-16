#!/usr/bin/env python

"""
Example script for testing quality of trained VxmDense models. This script iterates over a list of
images pairs, registers them, propagates segmentations via the deformation, and computes the dice
overlap. Example usage is:

    test.py  \
        --model model.h5  \
        --pairs pairs.txt  \
        --img-suffix /img.nii.gz  \
        --seg-suffix /seg.nii.gz

Where pairs.txt is a text file with line-by-line space-seperated registration pairs.
This script will most likely need to be customized to fit your data.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import time
import numpy as np
# import tensorflow as tf
import torch
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
VIS = False
if VIS:
    import matplotlib.pyplot as plt


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--model', required=True, help='VxmDense model file')
parser.add_argument('--pairs', required=True, help='path to list of image pairs to register')
parser.add_argument('--img-suffix', help='input image file suffix')
parser.add_argument('--seg-suffix', help='input seg file suffix')
parser.add_argument('--img-prefix', help='input image file prefix')
parser.add_argument('--seg-prefix', help='input seg file prefix')
parser.add_argument('--labels', help='optional label list to compute dice for (in npy format)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--compative', action='store_true', help='compative mode')
parser.add_argument('--seg_change_test', action='store_true', help='seg change test')
args = parser.parse_args()

# sanity check on input pairs
# if args.img_prefix == args.seg_prefix and args.img_suffix == args.seg_suffix:
#     print('Error: Must provide a differing file suffix and/or prefix for images and segs.')
#     exit(1)
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# device handling
# device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load seg labels if provided
labels = np.load(args.labels) if args.labels else None

# check if multi-channel data
add_feat_axis = not args.multichannel

# keep track of all dice scores
reg_times = []
dice_means = []

# with tf.device(device):
with torch.no_grad():
    # load model and build nearest-neighbor transfer model
    if not args.compative:
        model = vxm.networks.VxmDense.load(args.model, device = device)
    else:
        model = vxm.networks.VxmComp.load(args.model, device = device)
    model.to(device)
    model.eval()
    model.transformer.mode = 'nearest'

    # moved, warp = model(input_moving, input_fixed, registration=True)
    # registration_model = model.get_registration_model()
    # transform_model = model.transformer

    for i in range(len(img_pairs)):

        # load moving image and seg
        moving_vol = vxm.py.utils.load_volfile(
            img_pairs[i][0], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][0], np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)

        # load fixed image and seg
        fixed_vol = vxm.py.utils.load_volfile(
            img_pairs[i][1], np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed_seg = vxm.py.utils.load_volfile(
            seg_pairs[i][1], np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_vol = torch.from_numpy(moving_vol).to(device).float().permute(0, 3, 1, 2)
        moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 3, 1, 2)
        fixed_vol = torch.from_numpy(fixed_vol).to(device).float().permute(0, 3, 1, 2)
        fixed_seg = torch.from_numpy(fixed_seg).to(device).float().permute(0, 3, 1, 2)
        if args.seg_change_test:
            masked_lable_id = torch.randint(1, 24, moving_vol[:,:,0:1,0:1].shape, device=moving_vol.device)
            if torch.rand([1])<0.5:
                seg_change = True
            else:
                seg_change = False
            if torch.rand([1])<0.5 and seg_change:
                moving_vol[moving_seg==masked_lable_id] = 0.
            elif seg_change:
                fixed_vol[fixed_seg==masked_lable_id] = 0.

        # fixed_seg = [torch.from_numpy(d).to(device).float().unsqueeze(0).permute(0, 3, 1, 2) for d in fixed_seg]
        # fixed_seg = torch.from_numpy(fixed_seg).to(device)
        # predict warp and time
        start = time.time()
        reg, warp = model(moving_vol, fixed_vol, moving_seg, fixed_seg, registration=True)
        reg_time = time.time() - start
        if i != 0:
            # first keras prediction is generally rather slow
            reg_times.append(reg_time)

        # apply transform
        warped_seg = model.transformer(moving_seg, warp).squeeze()
        if VIS:
            # plt.imshow(moving_vol.cpu().numpy()[0, 0, :, :], cmap='gray')
            # plt.show()
            # plt.imshow(fixed_vol.cpu().numpy()[0, 0, :, :], cmap='gray')
            # plt.show()
            # plt.imshow(reg.cpu().numpy()[0, 0, :, :], cmap='gray')
            # plt.show()
            plt.imshow(moving_seg.cpu().numpy()[0, 0, :, :], cmap='gray')
            plt.colorbar()
            plt.show()
            plt.imshow(fixed_seg.cpu().numpy()[0, 0, :, :], cmap='gray')
            plt.colorbar()
            plt.show()
            plt.imshow(warped_seg.cpu().round().numpy(), cmap='gray')
            plt.colorbar()
            plt.show()
        # compute volume overlap (dice)
        overlap = vxm.py.utils.dice(warped_seg.cpu().round().numpy(), 
                                    fixed_seg.cpu().numpy()[0, 0, :, :], 
                                    labels=labels)
        dice_means.append(np.mean(overlap))
        print('Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (i + 1, reg_time,
                                                                    np.mean(overlap),
                                                                    np.std(overlap)))

print()
print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                    np.std(reg_times)))
print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))
