import pandas as pd
import os
import time


def output_csv_log(cfg, output_dir, train_loss, test_loss, best_perf, epoch, gflops):
    model_name = cfg.MODEL.NAME
    extra = cfg.MODEL.EXTRA
    if extra.USE_REGRESS_BRANCH and extra.USE_HEATMAP_BRANCH:
        output_type = 'regress_heatmap'
        loss_type = ('_').join([cfg.LOSS.CRITERION_REGRESS, cfg.LOSS.CRITERION_HEATMAP])
        sigma = cfg.MODEL.FACE_SIGMA
    if extra.USE_REGRESS_BRANCH:
        output_type = 'regress'
        loss_type = cfg.LOSS.CRITERION_REGRESS
        sigma = 0
    elif extra.USE_HEATMAP_BRANCH:
        output_type = 'heatmap'
        loss_type = cfg.LOSS.CRITERION_HEATMAP
        sigma = cfg.MODEL.FACE_SIGMA
    else:
        output_type = ''
        loss_type = cfg.LOSS.CRITERION
        sigma = 0
    EN = 'True' if cfg.MODEL.HEATMAP_EN else 'False'
    input_size = str(cfg.MODEL.IMAGE_SIZE)
    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    gpu_num = len(cfg.GPUS)
    optimizer = cfg.TRAIN.OPTIMIZER
    if cfg.TRAIN.SCHEDULER == 'MultiStepLR':
        scheduler = 'Step {}'.format(str(cfg.TRAIN.LR_STEP))
    elif cfg.TRAIN.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = 'ReduceLROnPlateau(p{})'.format(cfg.TRAIN.SCHEDULER_PATIENCE)
    test_perf = best_perf
    train_perf = ''
    time_str = time.strftime('%Y%m%d%H%M')
    output_dict = {
        "ID": time_str,
        "Backbone": model_name,
        "Dataset": cfg.FACE_DATASET.DATASET,
        "Type": output_type,
        "Loss": loss_type,
        "Quality": '',
        "EN": EN,
        "Input Size": input_size,
        "Batch Size": batch_size,
        "GPU Num": gpu_num,
        "Sigma": sigma,
        "Epoch": epoch,
        "Optim": optimizer,
        "Scheduler": scheduler,
        "Init LR": cfg.TRAIN.LR,
        "Test Perf": best_perf,
        "Train Perf": '',
        "Test Loss": test_loss,
        "Train Loss": train_loss,
        'GFLOPs': gflops
    }
    output_dir = os.path.join(output_dir, "train_status.csv")
    df = pd.DataFrame(output_dict, index=[0])
    df.to_csv(output_dir, index=False)
