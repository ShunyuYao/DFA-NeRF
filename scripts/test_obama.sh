export CUDA_VISIBLE_DEVICES=0
python NeRFs/DFANeRF/run_nerf_com_trainExpLater.py --config dataset/obama/HeadNeRF_config_ba.txt \
    --last_dist=1e10 \
    --datadir dataset/obama \
    --concate_bg --N_rand=2048 --sample_rate=0 --i_print=100  --i_test_person=10000 --chunk=2048 \
    --win_size=16 --smo_size=4 --smo_torse_size 8 --train_together --i_weights=100000 \
    --all_speaker --sample_rate_mouth=0 --lrate_decay=500 --lrate=5e-4 --use_et_embed --nosmo_iters=300000 \
    --dim_signal=96 --dim_aud=96 --n_object=1 \
    --N_iters=600000 \
    --expname=obama_TrainExpLater_smoMix \
    --aud_file=obama_aud.pt \
    --use_deformation_field \
    --exp_file=obama_64_32.pt \
    --use_ba \
    --render_person \
    --noexp_iters 400000 \
    --resume dataset/train_together/obama_TrainExpLater_smoMix/280000.tar \
    --test_file transforms_val_ba.json \
    --render_video