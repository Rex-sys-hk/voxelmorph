python setup.py install

./scripts/torch/train.py \
--img-list data/train_list_comp_train.txt \
--model-dir models/output/FC \
--gpu 0 \
--batch-size 2 \
--epochs 250 \
--steps-per-epoch 1 \
--compative \
# --initial-epoch 0 \
# --load-weights models/output/comp/0250.pt \
