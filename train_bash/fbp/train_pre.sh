# export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--task_name "Bfcn" \
--n_class 25 \
--mode 1 \
--dataset 'fbp' \
--batch_size 6 \
--sub_batch_size 32 \
--size_p 508 \
--size_g 508 \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/fbp \
--log_path ../../runs/fbp \
--pre_path ./ \
--start 0 \
--lens 100 \