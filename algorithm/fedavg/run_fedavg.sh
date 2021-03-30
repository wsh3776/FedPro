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


python fedavg_main.py --dataset demo --model cnn --lr 0.03 --batch_size 4 --client_num_in_total 14 --client_num_per_round 4 --num_rounds 12 --seed 2 --epoch 2 --eval_interval 1 --device cpu --note run_1_seed