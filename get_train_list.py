# write a list of OASIS subjects to a training file
import pathlib
path = pathlib.Path('/home/rxin/Downloads/tmp/voxelmorph/data/OASIS')
subj_lst_m = [str(f/'slice_norm.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
with open('data/train_list.txt','w') as tfile:
	tfile.write('\n'.join(subj_lst_m))
# compative
mask_lst_m = [str(f/'slice_seg24.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
sub_mask_lst = []
for i in range(len(subj_lst_m)):
	sub_mask_lst.append(f'{subj_lst_m[i]} {mask_lst_m[i]}')
with open('data/train_list_comp.txt','w') as tfile:
	tfile.write('\n'.join(sub_mask_lst))