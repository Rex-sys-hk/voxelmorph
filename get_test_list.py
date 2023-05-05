# write a list of OASIS subjects to a training file
import pathlib
import numpy as np
path = pathlib.Path('/home/rxin/Downloads/tmp/voxelmorph/data/OASIS_2d')
subj_lst_m = [str(f/'slice_') for f in path.iterdir() if str(f).endswith('MR1')]
lab_lst_m = [str(f/'slice_norm.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
pair_lst = []
for i in range(len(subj_lst_m)):
	pair_lst.append(f'{subj_lst_m[i]} {subj_lst_m[-i]}')

with open('data/test_list.txt','w') as tfile:
	tfile.write('\n'.join(pair_lst))
