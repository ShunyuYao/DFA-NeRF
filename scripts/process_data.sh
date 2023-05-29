export CUDA_VISIBLE_DEVICES=0
python data_util/process_data_ba.py --id=$1 --step=-1
python data_util/process_data_ba.py --id=$1 --step=0 &
python data_util/process_data_ba.py --id=$1 --step=1 --dst_size 512
python data_util/process_data_ba.py --id=$1 --step=2 --no_use_opFlow4FaceAlign --smooth_lms
python data_util/process_data_ba.py --id=$1 --step=3
python data_util/process_data_ba.py --id=$1 --step=4
python data_util/process_data_ba.py --id=$1 --step=5
# wait
python data_util/process_data_ba.py --id=$1 --step=6
python data_util/process_data_ba.py --id=$1 --step=7
python data_util/process_data_ba.py --id=$1 --step=8
python data_util/process_data_ba.py --id=$1 --step=9
python data_util/process_data_ba.py --id=$1 --step=10
python data_util/process_data_ba.py --id=$1 --step=11