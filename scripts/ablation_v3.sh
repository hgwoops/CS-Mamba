#!/bin/bash
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-30}
C="--is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --d_model 256 --d_state 16 --train_epochs $EPOCHS --batch_size 32 --learning_rate 0.0001 --itr 1 --des Exp --gpu $GPU"

echo "=== CS-Mamba V3 Ablation ==="

echo ""; echo ">>> [1/4] S-Mamba baseline <<<"
python -u run.py $C --model Mamba_simple --model_id base 2>&1 | grep -E "mse|mae" | tail -1

echo ""; echo ">>> [2/4] V2 DVR only <<<"
python -u run.py $C --model CSMamba --model_id v2 --use_ms_scan 0 --use_dvr 1 --use_freq_init 0 2>&1 | grep -E "mse|mae" | tail -1

echo ""; echo ">>> [3/4] V3 no-BiMamba <<<"
python -u run.py $C --model CSMambaV3 --model_id v3_nobi --bidirectional 0 --n_proto 64 2>&1 | grep -E "mse|mae" | tail -1

echo ""; echo ">>> [4/4] V3 full <<<"
python -u run.py $C --model CSMambaV3 --model_id v3_full --bidirectional 1 --n_proto 64 2>&1 | grep -E "mse|mae" | tail -1

echo ""; echo "=== Done ==="
