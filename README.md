# DFA-NeRF
Official implentation of [DFA-NeRF: Personalized Talking Head Generation via Disentangled Face Attributes Neural Rendering](https://zerzerzerz.github.io/DFA-NeRF/).


## Prerequisites
- You can create an anaconda environment called adnerf with:
    ```
    conda env create -f environment.yml
    conda activate adnerf
    ```
- Download the weights of models for data preprocessing and put them into corresponding positions in `data_util` folder.
    - [AliyunDrive](https://www.aliyundrive.com/s/AfvdwFom2RF), Code: vf40
    - [GoogleDrive](https://drive.google.com/drive/folders/1BHxGTi0b_A3zcTfgfavwgZE-VNts9j_E?usp=sharing)


## Train
- Data Preprocess ($id Obama for example)
    ```
    bash scripts/process_data.sh obama
    ```
    - Input: A portrait video containing voice audio. (dataset/vids/$id.mp4)
    - Output: folder dataset/$id that contains all files for training

- Train the NeRFs
   ```
   bash scripts/train_obama.sh
   ```

## Test
Run the following the command to test the trained models:
  ```
  bash scripts/test_obama.sh
  ```

## To Do List
- [ ] Release codes of Transformer GP-VAE proposed in our paper.
- [ ] Release codes for testing with your own speech files. Actually you can use the codes in `data_util/wav2exp/test_w2l_audio.py` to generate the aud file. 

## Citation
```
@article{yao2022dfa,
  title={DFA-NeRF: Personalized Talking Head Generation via Disentangled Face Attributes Neural Rendering},
  author={Yao, Shunyu and Zhong, RuiZhe and Yan, Yichao and Zhai, Guangtao and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2201.00791},
  year={2022}
}

## Acknowledgments
Most of the codes are referred to [AD-NeRF](https://github.com/YudongGuo/AD-NeRF).