./scripts/torch/register.py \
--moving data/OASIS_2d/OASIS_OAS1_0060_MR1/slice_norm.nii.gz \
--fixed data/OASIS_2d/OASIS_OAS1_0001_MR1/slice_norm.nii.gz \
--moved data/warped.nii.gz \
--model models/output/torch/0060.pt \
--gpu 0
