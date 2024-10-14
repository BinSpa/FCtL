# export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--task_name "all" \
--n_class 25 \
--mode 3 \
--dataset 'fbp' \
--batch_size 3 \
--sub_batch_size 8 \
--size_p 508 \
--size_g 508 \
--glo_path_10 "../../saved_models/fbp/fbpB10.epoch49.pth" \
--glo_path_15 "../../saved_models/fbp/fbpB15.epoch39.pth" \
--pre_path "../../saved_models/fbp/fbpBfcn.epoch29.pth" \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/fbp/ \
--log_path ../../runs/fbp \
--start 0 \
--lens 50 \