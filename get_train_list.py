# write a list of OASIS subjects to a training file
import pathlib
path = pathlib.Path('/home/rxin/Downloads/tmp/voxelmorph/data/OASIS')
subj_lst_m = [str(f/'slice_norm.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
with open('data/train_list.txt','w') as tfile:
	tfile.write('\n'.join(subj_lst_m))