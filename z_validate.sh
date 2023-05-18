python setup.py install
python ./scripts/torch/test.py \
--pairs data/test_list.txt \
--img-suffix norm.nii.gz \
--seg-suffix seg24.nii.gz \
--model models/output/FC/best.pt \
--compative \
--seg_change_test \
# --model models/pretrained/vxm_dense_brain_T1_3D_mse.h5 \

# python ./scripts/tf/test.py \
# --pairs data/test_list.txt \
# --img-suffix norm.nii.gz \
# --seg-suffix seg24.nii.gz \
# --model models/output/tf/1500.h5 \