python train.py --num_epochs=100 --num_workers=4 --resume --finetune_after=41 \
--model_name im2ingr --batch_size 76 --ingrs_only \
--es_metric iou_sample --loss_weight 0 1000.0 1.0 1.0 \
--learning_rate 1e-4 --scale_learning_rate_cnn 1.0 \
--save_dir ../checkpoints --recipe1m_dir ../data

python train.py --model_name model --batch_size 76 --recipe_only --transfer_from im2ingr \
--save_dir ../checkpoints --recipe1m_dir ../data