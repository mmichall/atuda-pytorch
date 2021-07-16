#!/bin/sh
python train.py --data_set 'amazon' --src_domain 'books' --tgt_domain 'kitchen' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 0.1e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'books' --tgt_domain 'electronics' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 0.1e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'books' --tgt_domain 'dvd' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'kitchen' --tgt_domain 'books' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'kitchen' --tgt_domain 'electronics' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'kitchen' --tgt_domain 'dvd' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'electronics' --tgt_domain 'books' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'electronics' --tgt_domain 'kitchen' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'electronics' --tgt_domain 'dvd' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'dvd' --tgt_domain 'books' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'dvd' --tgt_domain 'kitchen' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
python train.py --data_set 'amazon' --src_domain 'dvd' --tgt_domain 'electronics' --model 'AutoEncoder' --max_epochs 400 --train_batch_size 8 --learning_rate 1.0e-03 --learning_rate_kl 1.0e-03  --reduce_lr_factor 0.5 --denoising_factor 0.7 --epochs_no_improve 5 --autoencoder_shape '(5000, 3000)' --kl_threshold 999
