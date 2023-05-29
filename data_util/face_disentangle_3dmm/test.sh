# python test_model.py --ckpt_path logs/train_LSR2_data0725_dim20_8_202107251748 \
#     --dim_o 20 \
#     --dim_m 8

# python test_model.py --ckpt_path logs/train_LSR2_data0725_dim32_16_202107251617 \
#     --dim_o 32 \
#     --dim_m 16

CUDA_VISIBLE_DEVICES=1, python test_model.py --ckpt_path logs/train_LSR2_data0725_dim64_32_202107251918 \
    --dim_o 64 \
    --dim_m 32

# python test_model_sjtu.py --ckpt_path logs/train_LSR2_data0725_dim32_16_202107251617 \
#     --dim_o 32 \
#     --dim_m 16

# CUDA_VISIBLE_DEVICES=1, python test_model.py --ckpt_path logs/train_LSR2_data0725_dim32_16_202107251617 \
#     --dim_o 32 \
#     --dim_m 16
