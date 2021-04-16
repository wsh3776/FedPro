#!/bin/bash
# python fedavg_main1.py \
      # --dataset "demo" \
      # --model "cnn" \
      # --lr 0.03 \
      # --batch_size 4 \
      # --client_num_in_total 14 \
      # --client_num_per_round 4 \
      # --num_rounds 12 \
      # --seed 2 \
      # --epoch 2 \
      # --eval_interval 1 \
      # --device "cpu" \
      # --note "run_1_seed"
# mnist centralized
#python fedavg_main.py --note test --dataset mnist --model cnn_mnist --lr 0.001 --client_num_in_total 200 --client_num_per_round 20 --partition_method centralized --num_rounds 150 --batch_size 32 --seed 2 --epoch 2 --eval_interval 1 --wandb_mode run
# mnist homo
#python fedavg_main.py --note test --dataset mnist --model cnn_mnist --lr 0.001 --client_num_in_total 200 --client_num_per_round 20 --partition_method homo --num_rounds 150 --batch_size 32 --seed 2 --epoch 2 --eval_interval 1 --wandb_mode run
# mnist hetero
python fedavg_main.py --note test --dataset mnist --model cnn_mnist --lr 0.001 --client_num_in_total 200 --client_num_per_round 20 --partition_method hetero --num_rounds 150 --batch_size 32 --seed 2 --epoch 2 --eval_interval 1 --wandb_mode run