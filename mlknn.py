#@Time      :2018/9/14 14:27
#@Author    :zhounan
# @FileName: mlknn.py
import numpy as np
import pandas as pd
'''
#######################################################################################没问题
def dis(nda1,nda2):##计算两个属性之间的距离
    nda1=set(nda1);nda2 =set(nda2)
    if -1.0 in nda2:
        nda2.remove(-1)
    if -1.0 in nda1:
        nda1.remove(-1)
    return len(nda1^nda2 )

def knn(train_x, t_index, k):##寻找离第t个最近的k个元素的序号
    data_num = train_x.shape[0]##第一个维度的个数
    neighbors = np.zeros(k)##类别个数

    dist = [dis(train_x [i],train_x[t_index ]) for i in range(data_num) ]
    dist[t_index ]=float('inf')
    for i in range(k):
        temp = float('inf')
        temp_j = 0
        for j in range(data_num):
            if (j != t_index) and (dist[j] < temp):
                temp = dist[j]
                temp_j = j##最后的temp是最短距离,temp_j是最短距离的点的序号
        dist[temp_j] = float('inf')##该点已经被加进neighbors了，所以不能是最短的了，即变成了inf
        neighbors[i] = temp_j

    return neighbors

def evaluation(test_y,predict):##评估模型好坏，数据的导入与输出
    test_y = test_y.astype(np.int)
    print('------','test_y','------','\n',test_y)
    dfy=pd.DataFrame (test_y )
    dfpre=pd.DataFrame (predict )
    dfy.to_csv(r"C:/Users/86159/PycharmProjects/MLKNN/data/test_y.csv")
    dfpre.to_csv(r"C:/Users/86159/PycharmProjects/MLKNN/data/predict.csv")
    print('\n')
    print('------','predict','------','\n',predict )
    hamming_loss = HammingLoss(test_y, predict)
    print('hamming_loss = ', hamming_loss)

def HammingLoss(test_y, predict):##
    label_num = test_y.shape[1]
    test_data_num = test_y.shape[0]
    hamming_loss = 0
    temp = 0
    for i in range(test_data_num):
        ss=np.sum(test_y[i] ^ predict[i])
        temp = temp +ss#按位异或运算符，a=60=0011 1100 b=13=0000 1101 → a^b=0011 0001=49，因为数据只有0和1，所以可以表示对称差,

    hamming_loss = temp / label_num / test_data_num

    return hamming_loss

#######读取文件
train_x = np.loadtxt('dataset/train_yasuo_x.txt')
print('------','train_x1','------','\n',train_x )
#print(type(train_x ))
train_y = np.loadtxt('dataset/train_yasuo_y.txt')
print('------','train_y','------','\n',train_y )


#######初始化
k = 10
s = 1
label_num = train_y.shape[1]##第二行的列数
train_data_num = train_x.shape[0]
Ph1 = np.zeros(label_num)
Ph0 = np.zeros(label_num)
Peh1 = np.zeros([label_num, k + 1])#+1:0个，1个，，，，，k个，所以要加一
Peh0 = np.zeros([label_num, k + 1])


########
#计算先验概率duiduiduiduiduiduidui
for i in range(label_num):  ##对每一个标签而言
    cnt = 0  ##计数参量
    for j in range(train_data_num):  ##对于每一个示例而言
        if train_y[j][i] == 1:  ##第j个示例有第i个标签########需要改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cnt = cnt + 1  ##循环结束之后是，有i标签的示例的个数
    Ph1[i] = (s + cnt) / (s * 2 + train_data_num)  ##计算ph1，i.e.一个示例有i标签的概率
    Ph0[i] = 1 - Ph1[i]  ##ph0，i.e.一个示例没有i标签的概率


for i in range(k):  ##对于每一个标签
    print('training for label\n', i + 1)
    c1 = np.zeros(k + 1)  ##c[]，初始化
    c0 = np.zeros(k + 1)  ##c'[],初始化
    for j in range(train_data_num):  ##对每一个示例
        temp = 0
        neighbors = knn(train_x, j, k)  ##找到该示例的邻集

        for l in range(k):  ##对每一个邻居而言,为的是求k个邻居中的有几个有这个i标签
            temp = temp + int(train_y[int(neighbors[l])][i])  ##循环结束之后temp是t的邻集中有i标签的个数

        if train_y[j][i] == 1:  ##判断t有没有i标签
            c1[temp] = c1[temp] + 1
        else:
            c0[temp] = c0[temp] + 1

    for j in range(k + 1):
        Peh1[i][j] = (s + c1[j]) / (s * (k + 1) + np.sum(c1))  ##计算peh1，i.e.t有i标签的情况下，邻集有j个含有i标签的概率
        Peh0[i][j] = (s + c0[j]) / (s * (k + 1) + np.sum(c0))

test_x = np.loadtxt('dataset/test_yasuo_x.txt')
test_y = np.loadtxt('dataset/test_yasuo_y.txt')
predict = np.ones(test_y.shape, dtype=np.int)
test_data_num = test_x.shape[0]##读取数据

for i in range(test_data_num):##对于每个示例而言
    neighbors = knn_test(train_x, test_x[i], k)##寻找第i个示例的最近的k个示例的index
    for j in range(label_num):##对于每个标签而言
        temp = 0
        for nei in neighbors:##nei就是index
            temp = temp + int(train_y[int(nei)][j])
        if(Ph1[j] * Peh1[j][temp] > Ph0[j] * Peh0[j][temp]):##P(Elj,Hl1)>P(Elj,Hl0),i.e.示例有i，有j个邻居有i同时发生的概率,>,示例没有i，有j个邻居有i同时发生的概率
            predict[i][j] = 1
        else:
            predict[i][j] = 0
#np.save('parameter_data/predict.npy', predict,allow_pickle=True)
print(predict )
evaluation(test_y,predict)'''
import numpy as np
import pandas as pd

#######################################################################################没问题
def dis(nda1,nda2):##计算两个属性之间的距离
    nda1=set(nda1);nda2 =set(nda2)
    if -1.0 in nda2:
        nda2.remove(-1)
    if -1.0 in nda1:
        nda1.remove(-1)
    return len(nda1^nda2 )

def knn(train_x, t_index, k):##寻找离第t个最近的k个元素的序号
    data_num = train_x.shape[0]##第一个维度的个数
    neighbors = np.zeros(k)##类别个数

    dist = [dis(train_x [i],train_x[t_index ]) for i in range(data_num) ]
    dist[t_index ]=float('inf')
    for i in range(k):
        temp = float('inf')
        temp_j = 0
        for j in range(data_num):
            if (j != t_index) and (dist[j] < temp):
                temp = dist[j]
                temp_j = j##最后的temp是最短距离,temp_j是最短距离的点的序号
        dist[temp_j] = float('inf')##该点已经被加进neighbors了，所以不能是最短的了，即变成了inf
        neighbors[i] = temp_j

    return neighbors

def evaluation(test_y,predict):##评估模型好坏，数据的导入与输出
    test_y = test_y.astype(np.int)
    print('------','test_y','------','\n',test_y)
    print('\n')
    print('------','predict','------','\n',predict )
    hamming_loss=0
    for i in range(test_y .shape [1]):
        hamming_loss += dis(test_y[i], predict[i])
    hamming_loss=hamming_loss/test_y .shape [1]/len(test_y [0])
    print('hamming_loss = ', hamming_loss)

#######读取文件
train_x = np.loadtxt('dataset/train_yasuo_x.txt')
print('------','train_x','------','\n',train_x )
#print(type(train_x ))
train_y = np.loadtxt('dataset/train_yasuo_y.txt')
print('------','train_y','------','\n',train_y )


#######初始化
k = 10
s = 1
label_num = train_y.shape[1]##第二行的列数
train_data_num = train_x.shape[0]
Ph1 = np.zeros(label_num)
Ph0 = np.zeros(label_num)
Peh1 = np.zeros([label_num, k + 1])#+1:0个，1个，，，，，k个，所以要加一
Peh0 = np.zeros([label_num, k + 1])


########
#计算先验概率
for i in range(label_num):  ##对每一个标签而言
    cnt = 0  ##计数参量
    for j in range(train_data_num):  ##对于每一个示例而言
        if i in train_y[j]:  ##第j个示例有第i个标签,判断j示例有没有i标签
            cnt = cnt + 1  ##循环结束之后是，有i标签的示例的个数
    Ph1[i] = (s + cnt) / (s * 2 + train_data_num)  ##计算ph1，i.e.一个示例有i标签的概率
    Ph0[i] = 1 - Ph1[i]  ##ph0，i.e.一个示例没有i标签的概率

for i in range(label_num):  ##对于每一个标签
    print('training for label\n', i + 1)
    c1 = np.zeros(k + 1)  ##c[]，初始化
    c0 = np.zeros(k + 1)  ##c'[],初始化

    for j in range(train_data_num):  ##对每一个示例
        temp = 0
        neighbors = knn(train_x, j, k)  ##找到该示例的邻集
        for l in range(k):  ##对每一个邻居而言,为的是求k个邻居中的有几个有这个i标签
            if i in train_y[int(neighbors[l])]:
                temp += 1  ##循环结束之后temp是t的邻集中有i标签的个数

        if i in train_y[j]:  ##判断t有没有i标签
            c1[temp] = c1[temp] + 1
        else:
            c0[temp] = c0[temp] + 1

    for j in range(k + 1):
        Peh1[i][j] = (s + c1[j]) / (s * (k + 1) + np.sum(c1))  ##计算peh1，i.e.t有i标签的情况下，邻集有j个含有i标签的概率
        Peh0[i][j] = (s + c0[j]) / (s * (k + 1) + np.sum(c0))

test_x = np.loadtxt('dataset/test_yasuo_x.txt')
test_y = np.loadtxt('dataset/test_yasuo_y.txt')
predict = [list(np.ones(test_y.shape[1],dtype=np.int))]

test_data_num = test_x.shape[0]##读取数据
for i in range(test_data_num):##对于每个示例而言
    temp0=test_x[i]
    temp0=np.append(temp0,-1)
    train_x[0]=temp0
    neighbors = knn(train_x, 0, k)##寻找第i个示例的最近的k个示例的index
    for j in range(label_num):##对于每个标签而言
        temp = 0
        temp1=[]
        c=[]
        for nei in neighbors:##nei就是index
            if j in train_y[int(nei)]:
                temp =temp+ 1  ##循环结束之后temp是t的邻集中有j标签的个数
        if(Ph1[j] * Peh1[j][temp] > Ph0[j] * Peh0[j][temp]):##P(Elj,Hl1)>P(Elj,Hl0),i.e.示例有i，有j个邻居有i同时发生的概率,>,示例没有i，有j个邻居有i同时发生的概率
            temp1.append(j)
    predict.append(temp1)

predict =pd.DataFrame (predict[1:] )
predict =predict .values
#np.save('parameter_data/predict.npy', predict,allow_pickle=True)
print(predict )
dfy=pd.DataFrame (test_y )
dfpre=pd.DataFrame (predict )
dfy.to_csv(r"C:/Users/86159/PycharmProjects/MLKNN/data/test_y.csv")
dfpre.to_csv(r"C:/Users/86159/PycharmProjects/MLKNN/data/predict.csv")
evaluation(test_y,predict)