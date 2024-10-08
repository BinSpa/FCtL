# export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--task_name "Bfcn" \
--n_class 6 \
--mode 1 \
--dataset 'gid' \
--batch_size 6 \
--sub_batch_size 32 \
--size_p 508 \
--size_g 508 \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/gid/ \
--log_path ../../runs/gid \
--pre_path ../../saved_models/gid/ \
--start 0 \
--lens 50 \