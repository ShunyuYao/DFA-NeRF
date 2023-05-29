export CUDA_VISIBLE_DEVICES=1
python demo_face_eye_detectPerframe_save.py --cfg experiments/300w_lp_menpo2D/hrnet_hm.yaml\
    --cfg_eye experiments/eye_300w_menpo/ghostnet_en_de.yaml \
    --testModelPath ./models/face_lms_68kpts_hrnet.pth \
    --testEyeModelPath ./models/eye_lms_6kpts.pth \
    --inputPath /home/yaosy/projects/dfa-ne-rf/dataset/train_together/test_Trump000006_audFanRemoveSlience_headonly_smoMix/video_test_Trump000006_audFanRemoveSlience_headonly_smoMix_aud_head.mp4 \
    --outputVidPath /home/yaosy/projects/dfa-ne-rf/dataset/train_together/test_Trump000006_audFanRemoveSlience_headonly_smoMix \
    --outputSavePath /home/yaosy/projects/dfa-ne-rf/dataset/train_together/test_Trump000006_audFanRemoveSlience_headonly_smoMix \
    --testMode video \
    --eye_heatmap_decode \
    --face_type 300W \
    --use_optical_flow