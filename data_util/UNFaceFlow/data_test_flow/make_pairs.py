import os

data_path = "/data/yudong/TalkingHead-NeRF/data_obama/obama_crop/"
mask_path = "/data/yudong/TalkingHead-NeRF/data_obama/obama_mask/"
fout = open("obama.txt", "w")
for i in range(11000):
    fout.write(data_path+str(18)+".jpg " + mask_path+str(18)+".jpg " + data_path+str(i)+".jpg " + mask_path+str(i)+".jpg\n")
