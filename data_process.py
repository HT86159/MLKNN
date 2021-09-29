#@Time      :2018/9/14 13:39
#@Author    :zhounan
# @FileName: data_process.py

import pandas as pd
import numpy as np
from scipy.io import arff

train_file_path = 'yeast_corpus/yeast-train.arff'#获取训练集路径
test_file_path = 'yeast_corpus/yeast-test.arff'

train_data, meta = arff.loadarff(train_file_path)  #读取训练集数据,返回的第一个数据是array，第二个数据是其他的文件信息
test_data, meta = arff.loadarff(test_file_path)
#print('train_data:','\n',train_data )

train_df = pd.DataFrame(train_data)       #因为array操作并不方便，所以转换为df
test_df = pd.DataFrame(test_data)
#print('train_df:','\n',train_df )

train_data = train_df.values                #values为ndarray
test_data = test_df.values
#print("tran_data:",'\n',train_data )


train_x = np.array(train_data[:, 0:103])     #所有行、第1-103列,需要自己去数多少行
train_y = np.array(train_data[:, 104:117])
test_x = np.array(test_data[:, 0:103])
test_y = np.array(test_data[:, 104:117])

def save(path,data):
    df=pd.DataFrame(data)
    df.to_excel(path)

#save('dataset/train_x.xlsx', train_x)
np.savetxt('dataset/train_x1.txt', train_x)
#save('dataset/test_x.xlsx', test_x)
np.savetxt('dataset/test_x1.txt', test_x)
##上次把dataset中的数据用xlsx存贮了
##下次应该把剩余的数据也用xlsx存储