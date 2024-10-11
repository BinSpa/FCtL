# export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--task_name "B15" \
--n_class 25 \
--mode 2 \
--dataset 'fbp' \
--batch_size 3 \
--sub_batch_size 8 \
--size_p 508 \
--size_g 508 \
--context10 3 \
--pre_path ../../saved_models/fbp/fbpBfcn.epoch29.pth \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/fbp/ \
--log_path ../../runs/fbp \
--start 0 \
--lens 50 \
