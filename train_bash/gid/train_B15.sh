# export CUDA_VISIBLE_DEVICES=0
python train_deep_globe.py \
--task_name "B15" \
--n_class 6 \
--mode 2 \
--dataset 'gid' \
--batch_size 6 \
--sub_batch_size 16 \
--size_p 508 \
--size_g 508 \
--context10 3 \
--pre_path ../../saved_models/gid/gidBfcn.epoch69.pth \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/gid/ \
--log_path ../../runs/gid \
--start 0 \
--lens 50 \
