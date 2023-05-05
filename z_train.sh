./scripts/torch/train.py \
--img-list data/train_list.txt \
--model-dir models/output/comp \
--gpu 0 \
--batch-size 96 \
--epochs 200 \
--initial-epoch 0 \
--steps-per-epoch 1 \
# --load-model models/output/1500.pt \
