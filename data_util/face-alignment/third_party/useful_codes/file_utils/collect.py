#coding:utf-8
#放置在pixiv目录下：boys, girls, ****.py
import os
import shutil
import json

for j in ['boys', 'girls']:
    path = './' + j
    out_path = './' + j + '_photos'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    listdir = os.walk(path)
    num = 0
    num_move = 0
    dic = {}
    for root, dirs, files in listdir:

        for i in files:
            if len(i) > 0:
                if root[0] == '.':
                    filetype=os.path.splitext(i)[1]
                    nam =  '{:012d}'.format(num)

                    root = root[2:]

                    dic.setdefault(nam,root +'\\' + os.path.splitext(i)[0])
                    num+=1
                    file_path = os.path.join(root, i)
                    # print(root, nam)
                    os.rename(file_path, os.path.join(root,nam + filetype))
                    shutil.move(os.path.join(root, nam + filetype), out_path + '/')
                    #try:
                        #shutil.move(file_path, out_path + '/')
                    #except:
                        #os.rename(file_path, os.path.join(root, str(num) + "_" + i))
                        #shutil.move(os.path.join(root, str(num)+"_"+i), out_path + '/')
                        #num_move += 1
                    #else:
                        #num_move += 1
        if num_move%100 == 0 and num_move > 0:
            print('move ({}/{}) {} photos'.format(num_move, num, j))
    with open(j + '.json', 'w',encoding='utf-8') as f:
        json.dump(dic, f)
    # f = open(j + '.json', encoding='utf-8')    #打开文件
    # print(json.load(f))
print("down!")
