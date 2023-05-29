cd wav2exp
CUDA_VISIBLE_DEVICES=0, python test_w2l_audio.py --input_path /home/yaosy/projects/dfa-ne-rf/dataset/test_auds/fanzhiyi_wo_reporter_remove_silence.wav \
    --save_path /home/yaosy/projects/dfa-ne-rf/dataset/test_auds/fanzhiyi_wo_reporter_remove_silence.pt
cd ..