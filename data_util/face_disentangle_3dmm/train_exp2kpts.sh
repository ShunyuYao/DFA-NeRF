# python train_exp2kpts.py --exp_name  train_exp2kpts_hiddenLayers0 \
#     --bs 512 \
#     --exp_dim 8 \
#     --num_hidden_layers 0 \
#     --eps 20

# python train_exp2kpts.py --exp_name  train_exp2kpts_hiddenLayers1 \
#     --bs 512 \
#     --exp_dim 8 \
#     --num_hidden_layers 1 \
#     --eps 20

# python train_exp2kpts.py --exp_name  train_exp2kpts_hiddenLayers2 \
#     --bs 512 \
#     --exp_dim 8 \
#     --num_hidden_layers 2 \
#     --eps 20

# python train_exp2kpts.py --exp_name  train_exp2kpts_exp_dim16_wBN \
#     --bs 512 \
#     --exp_dim 16 \
#     --num_hidden_layers 1 \
#     --eps 20 \
#     --exp_train_path ./dataset/exp_m_dim16_train_0725.npy \
#     --kpts_train_path ./dataset/face3dmmAlignKpts_train_0725.npy \
#     --exp_val_path ./dataset/exp_m_dim16_val_0725.npy \
#     --kpts_val_path ./dataset/face3dmmAlignKpts_val_0725.npy

python train_exp2kpts.py --exp_name  train_exp2kpts_exp_dim32_wBN \
    --bs 512 \
    --exp_dim 32 \
    --num_hidden_layers 1 \
    --eps 20 \
    --exp_train_path ./dataset/exp_m_dim32_train_0725.npy \
    --kpts_train_path ./dataset/face3dmmAlignKpts_train_0725.npy \
    --exp_val_path ./dataset/exp_m_dim32_val_0725.npy \
    --kpts_val_path ./dataset/face3dmmAlignKpts_val_0725.npy