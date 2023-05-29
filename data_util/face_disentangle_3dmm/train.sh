# python -u train_with_cycle.py \
#     --exp_name train_all_withCycleVec \
#     --lr 0.001 \
#     --eps 500 \
#     --lr_step 300,400 \
#     --use_vec_loss

# python -u train_with_cycle_backOnce.py \
#     --exp_name train_all_withCycleVec_backwardOnce \
#     --lr 0.001 \
#     --eps 500 \
#     --lr_step 300,400 \
#     --use_vec_loss

# python -u train_with_cycle_backOnce.py \
#     --exp_name train_all_withCycleVec_backwardOnce_woVecLoss \
#     --lr 0.001 \
#     --eps 500 \
#     --lr_step 470
#     # --use_vec_loss

# CUDA_VISIBLE_DEVICES=0, python -u train_with_cycle_backOnce.py \
#     --exp_name train_all_withCycleVec_woVecLoss \
#     --bs 512 \
#     --num_workers 4 \
#     --lr 0.001 \
#     --eps 120 \
#     --lr_step 40,100

CUDA_VISIBLE_DEVICES=1, python -u train_with_cycle_backOnce.py \
    --exp_name train_LSR2_data0725_dim32_16 \
    --bs 512 \
    --num_workers 4 \
    --dim_o 32 \
    --dim_m 16 \
    --lr 0.001 \
    --eps 50 \
    --lr_step 40

CUDA_VISIBLE_DEVICES=1, python -u train_with_cycle_backOnce.py \
    --exp_name train_LSR2_data0725_dim20_8 \
    --bs 512 \
    --num_workers 4 \
    --dim_o 20 \
    --dim_m 8 \
    --lr 0.001 \
    --eps 50 \
    --lr_step 40

CUDA_VISIBLE_DEVICES=1, python -u train_with_cycle_backOnce.py \
    --exp_name train_LSR2_data0725_dim64_32 \
    --bs 512 \
    --num_workers 4 \
    --dim_o 64 \
    --dim_m 32 \
    --lr 0.001 \
    --eps 50 \
    --lr_step 40