# export CUDA_VISIBLE_DEVICES=0
python ../../train_deep_globe.py \
--task_name "all" \
--n_class 6 \
--mode 3 \
--dataset 'gid' \
--batch_size 6 \
--sub_batch_size 8 \
--size_p 508 \
--size_g 508 \
--glo_path_10 "../../saved_models/gid/gidB10.epoch69.pth" \
--glo_path_15 "../../saved_models/gid/gidB15.epoch24.pth" \
--pre_path "../../saved_models/gid/gidBfcn.epoch69.pth" \
--train \
--val \
--data_path /data1/gyl/RS_DATASET/FBP \
--model_path ../../saved_models/gid/ \
--log_path ../../runs/gid \
--start 0 \
--lens 50 \
