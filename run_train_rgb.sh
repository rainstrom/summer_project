#CUDA_VISIBLE_DEVICES=1 nice -n 19 python train_rgb_vgg16.py # | tee log_train_rgb_vgg16.txt
CUDA_VISIBLE_DEVICES=1 nice -n 19 python train_rgb_vgg16_finetuneconv.py #| tee log_train_rgb_vgg16_finetuneconv.txt
