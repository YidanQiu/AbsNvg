import pandas as pd
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, MeanShift, estimate_bandwidth
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import datasets, optimizers, metrics, layers, Sequential, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
import os
from save_to_excel import save_to_xlsx, make_excel, write_excel

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 这两行可以让plt显示中文


def reduction(X, n_components=2, init='pca'):  # TSNE数据降维，用pca初始化
    X_tsne = TSNE(n_components=n_components, init=init, random_state=0)
    X = X_tsne.fit_transform(X)
    return X


def prep4category(data, subj, dim):
    subj_list = data['participant']
    dim_list = data['dim']
    route_list = data['route']
    ACC_list = data['ACC']
    RT_list = data['RT']

    route_ind = []
    ACCs = []
    RTs = []
    for i in range(len(subj_list)):
        if subj_list[i] == subj and dim_list[i] == dim:
            route_ind.append(route_list[i])
            ACCs.append(ACC_list[i])
            RTs.append(RT_list[i])

    sorted_data = []
    n_route = 0
    route_len = []
    prev_route = 0
    num_route = -1  # 第几条路
    for r in range(len(route_ind)):
        if route_ind[r] != prev_route:
            sorted_data.append([[ACCs[r], RTs[r]]])
            num_route += 1
            if num_route > 0:
                route_len.append(n_route)
            n_route = 1  # 当前路径的选择数量
            prev_route = route_ind[r]
        else:
            sorted_data[num_route].append([ACCs[r], RTs[r]])
            n_route += 1
        route_len.append(n_route)

    max_step = max(route_len)
    for e in sorted_data:
        if len(e) < max_step:
            d = max_step - len(e)
            a = []
            rt = []
            for i in e:
                a.append(i[0])
                rt.append(i[1])
            for j in range(d):
                e.append([np.average(a), np.average(rt)])

    sorted_data = np.array(sorted_data)  # （路径数量，每条路径走了多少步，2)其中2个数值，一个是正确率，一个是反应时 (20,4,2)
    sorted_data = np.reshape(sorted_data, (num_route + 1, max_step * 2))  # (b,c)
    return sorted_data


def extract_label(subj, dim, data):
    sub_list = data['subj']
    dim_list = data['dim']
    route_list = data['route']
    label_list = data['label']

    labels = []
    for i in range(len(sub_list)):
        if sub_list[i] == subj and dim_list[i] == dim:
            labels.append(label_list[i])
    labels = np.array(labels)
    return labels


def K_means(X, y=None, n_clusters=10):  # K_means聚类
    model = KMeans(n_clusters=n_clusters, random_state=0)
    y_pred = model.fit_predict(X, y=y)
    return y_pred, model  # y_pred是kmeans预测结果


def minibatch_kmeans(X, y, n_clusters=10):  # minibatch_kmeans聚类
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    y_pred = model.fit_predict(X)
    return y_pred, model


def Mean_shift(X, y, n_clusters=10):  # Mean_shift聚类
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=0)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    y_pred = model.fit_predict(X)
    return y_pred, model


def DB_scan(X, y, eps=3, min_samples=10):  # DB_scan聚类
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X)
    return y_pred, model


def NetWork():
    # image_input = Input((h, w))
    model = Sequential([  # filters是卷积核的数量，kernel_size是卷积核的尺寸大小，strides是卷积步长，padding是是否需要边界填充
        layers.Dense(units=64, activation='relu'),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=2, activation='softmax')  # 输出层是全连接层
    ])
    # model.build(input_shape=(None, c))
    # model = Model(image_input, x)
    return model


def divide_data(X, label, n_test):
    temp = []  # 构建专属长度的列表，用于随机抽取测试数据
    for i in range(len(label)):
        temp.append(i)
    test_ind = np.random.choice(temp, n_test, replace=False)
    if n_test == 1:
        test_ind = [test_ind]

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for ind in temp:
        if ind in test_ind:
            x_test.append(X[ind])
            y_test.append(label[ind])
        else:
            x_train.append(X[ind])
            y_train.append(label[ind])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def metric(X, y_pred):  # calinski_harabaz评价指标，越大说明聚类效果越好
    calinski_harabasz_index = sklearn.metrics.calinski_harabasz_score(X, y_pred)
    return calinski_harabasz_index


def show_result(X, y_pred, model, result_image_path, show=True, title=r'聚类结果'):  # 显示聚类后结果
    if X.shape[-1] != 2:
        X = reduction(X)  # 降成2维:(b,c)-->(b,2)
    b = MinMaxScaler()  # 归一化为0到1，方便显示
    X = b.fit_transform(X)
    # print(X.shape, X.min(), X.max())
    # decision_region(X, classifier=model, input_dtype=X.dtype, show=show)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'coral', 'lime', 'brown']
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], s=30, color=colors[y_pred[i]])
        # plt.text(X[i, 0], X[i, 1], s=y_pred[i], color=colors[y_pred[i]])
    plt.title(title)
    plt.axis('off')
    plt.savefig(result_image_path)
    if show:
        plt.show()
    plt.close()


txt = r'C:\Users\QiuYiDan\Documents\practice_py\MyExp\DataAnalysis\学习项目\神经网络_kmeans分类\log.txt'
txt = open(txt, 'w')
data = pd.read_excel(r'all_data.xlsx') # sort all data in excel and input here
label_data = pd.read_excel(r'learning_label.xlsx') # input the corresponding labels
dim_list = [1, 2, 3]
subj_list = ['000', '001', '002', '003', '004', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
             '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028']
# subj_list = ['000', '001', '002', '003']
n_class = 2  # 类别数
encode = True  # 是否使用神经网络编码
random_choice = True  # 是否随机打乱训练测试集
clustering_method = 'kmeans'  # 聚类方式有：'kmeans' 'minibatch_kmeans' 'mean_shift' 'db_scan'

os.makedirs('checkpoint', exist_ok=True)
dir_path = r'result_xlsx/%s' % clustering_method
os.makedirs(dir_path, exist_ok=True)
calinski_harabasz_index_list = []
# test_list = [2, 3, 4, 5]  # 选取后几个样本进行测试，其余训练
test_list = [3]  # 选取后几个样本进行测试，其余训练
for test_num in test_list:
    y_pred_list = []
    save_result_list = []
    for dim in dim_list:  # 遍历任务
        for sub in subj_list:  # 遍历受试者
            sub = f'sub{sub}'
            X = prep4category(data, sub, dim)  # (b,c)
            y = extract_label(sub, dim, label_data)  # (b,)
            # print('dim:',dim,sub,X.shape,y.shape)
            b, c = X.shape
            if random_choice:  # 随机同时打乱X和y
                ids = np.random.choice(range(0, len(X)), size=(len(X)), replace=False)
                X = X[ids]
                y = y[ids]
            # train_num = len(X) - test_num
            x_train, x_test = X[0:-test_num, :], X[-test_num:, :]  # 划分数据
            y_train, y_test = y[0:-test_num], y[-test_num:]  # 划分标签
            y_train_onehot = tf.one_hot(y_train, depth=n_class)
            y_test_onehot = tf.one_hot(y_test, depth=n_class)
            # 搭建网络模型
            model = NetWork()
            model.build(input_shape=(None, c))  # 建立神经网络模型，None表示批次，属于固定写法
            # model.summary()  # 查看网络每一层结构
            save_weight_path = r'checkpoint/dim%s_%s_test%s_%s.h5' % (
                dim, sub, test_num, random_choice)  # 模型权重保存路径
            try:  # 尝试加载上次训练的模型
                model.load_weights(save_weight_path)
                # print('%s权重加载成功' % save_weight_path)
            except:
                # ----------直接训练网络----------
                # 设置训练和测试指标
                model.compile(optimizer=tf.optimizers.Adam(lr=0.001),  # 优化器
                              loss=tf.losses.CategoricalCrossentropy(),  # 交叉熵损失函数
                              metrics=['accuracy'])  # 准确率
                # 创建保存训练的网络权重的容器
                mc = ModelCheckpoint(filepath=save_weight_path, monitor='val_loss', verbose=2, save_best_only=True,
                                     save_weights_only=True)  # 模型权重保存到filepath里
                model.fit(x_train, y_train_onehot, epochs=100, validation_data=(x_test, y_test_onehot),
                          callbacks=[mc, ], verbose=0)
            # 测试网络，即把这个人的所有数据扔进神经网络进行编码为向量，然后进行kmeans聚类
            if encode:  # 数据经过模型编码
                X_cluster = model.predict(X)  # (b,h,w,c)-->(b,n_class)相当于每一张把图片编码为对应向量，我们对这向量进行聚类，而不是直接对原图聚类
                y_test_pred = model.predict(x_test)
                y_test_pred = np.argmax(y_test_pred, axis=-1)
                print('预测类别:', y_test_pred)
                print('真实类别:', y_test)
                print('准确率:', np.mean(np.equal(y_test_pred, y_test)))
                # print(y_pred)
                # print(y)
                # print('准确率:', np.mean(np.equal(y_pred, y)),file=txt)

                save_dir = r'result_image/%s/test%s_%s' % (
                    clustering_method, test_num, random_choice)
                os.makedirs(save_dir, exist_ok=True)  # 这里放经过神经网络编码后的数据聚类结果
                result_image_path = os.path.join(save_dir, r'dim%s_%s.png' % (dim, sub))  # 聚类结果保存路径
            else:
                X_cluster = X
                save_dir = 'result_image/original/test%s_%s' % (test_num, random_choice)
                os.makedirs(save_dir, exist_ok=True)  # 这里放原数据聚类结果
                result_image_path = os.path.join(save_dir, r'dim%s_%s.png' % (dim, sub))  # 聚类结果保存路径
            X_cluster = np.reshape(X_cluster, [X_cluster.shape[0], -1])  # (b,c)
            if clustering_method == 'kmeans':
                y_pred, model = K_means(X_cluster, n_clusters=n_class)
            elif clustering_method == 'minibatch_kmeans':
                y_pred, model = minibatch_kmeans(X_cluster, y, n_clusters=n_class)  # 进行minibatch_kmeans聚类
            elif clustering_method == 'mean_shift':
                y_pred, model = Mean_shift(X_cluster, y)  # 进行mean_shift聚类
            elif clustering_method == 'db_scan':
                y_pred, model = DB_scan(X_cluster, y, min_samples=n_class)  # 进行DB_scan聚类
            else:
                print('聚类方式输入有误，程序退出')
                exit()
            y_pred_list.append(y_pred)

            calinski_harabasz_index = metric(X_cluster, y_pred)  # calinski_harabaz评价指标，越大说明聚类效果越好
            calinski_harabasz_index_list.append(calinski_harabasz_index)
            print('受试者:%s' % sub, 'dim:%s' % dim, '样本数:%s' % len(y_pred),
                  '%s聚成%s类calinski_harabaz指数为:%.4f' % (clustering_method, n_class, calinski_harabasz_index)
                  , '类别标签:', np.unique(y_pred))
            print('受试者:%s' % sub, 'dim:%s' % dim, '样本数:%s' % len(y_pred),
                  '%s聚成%s类calinski_harabaz指数为:%.4f' % (clustering_method, n_class, calinski_harabasz_index)
                  , '类别标签:', np.unique(y_pred), file=txt)
            title = r'任务:%s 受试者:%s %s聚类 样本数:%s calinski_harabaz指数:%.4f' % (
                dim, sub, clustering_method, len(y_pred), calinski_harabasz_index)
            show_result(X_cluster, y_pred, model, result_image_path, show=False, title=title)
            print(sub, dim, y_pred)
            # 结果保存到xlsx
            save_result_list.append([dim, sub, y_pred, calinski_harabasz_index])
            save_name = os.path.join(dir_path, r'test%s_%s.csv' % (test_num, random_choice))
            results = pd.DataFrame(save_result_list, columns=['dim', 'subj', 'classified',
                                                              'calinski_harabasz'])  # save_result_list：(b,3)
            results.to_csv(save_name, index=False)  # 取出最左边的行索引
            print('已保存到:%s' % save_name)
