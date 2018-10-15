import numpy as np

#最小二乘法求解
def least_square(feature, label):
    '''
    最小二乘法
    :param feature: 特征
    :param label:  标签
    :return:  w(mat) 回归系数
    '''
    w = (feature.T * feature).I * feature.T * label
    return w

#Armijo搜索
#损失函数计算
def get_error(feature, label, w):
    '''
    计算误差
    :param feature: 特征
    :param label:  标签
    :param w: 参数
    :return:损失函数
    '''
    return (label - feature * w).T * (label - feature * w ) / 2


#最小m值得计算
def get_min_m(feature, label, sigma, delta, d, w, g):
    '''
    计算步长中最小的值m
    :param feature: 特征
    :param label: 标签
    :param sigma:   全局牛顿法的参数
    :param delta:  全局牛顿法的参数
    :param d: 负的一阶导数除以二阶导数值
    :param w:
    :param g: 一阶导数值
    :return: m 最小m值
    '''
    m = 0
    while True:
        w_new = w + pow(sigma, m) * d
        left = get_error(feature, label, w_new)
        right = get_error(feature, label, w) + delta * pow(sigma, m) * g.T * d
        if left <= right :
            break
        else:
            m +=1
    return m

#一阶导数
def first_derivativ(feature, label, w):
    '''计算一阶导函数的值
    input:  feature(mat):特征
            label(mat):标签
    output: g(mat):一阶导数值
    '''
    m, n = np.shape(feature)
    g = np.mat(np.zeros((n, 1)))
    for i in range(m):
        err = label[i, 0] - feature[i, ] * w
        for j in range(n):
            g[j, ] -= err * feature[i, j]
    return g


#二阶导数
def second_derivative(feature):
    '''计算二阶导函数的值
    input:  feature(mat):特征
    output: G(mat):二阶导数值
    '''
    m, n = np.shape(feature)
    G = np.mat(np.zeros((n, n)))
    for i in range(m):
        x_left = feature[i, ].T
        x_right = feature[i, ]
        G += x_left * x_right
    return G


#全局牛顿法
def newton(feature, label, iterMax, sigma, delta):
    '''
    牛顿法
    :param feature: 特征
    :param label:  标签
    :param iterMax:  最大迭代次数
    :param sigma:    参数
    :param delta:    参数
    :return:  w 回归系数
    '''
    n = np.shape(feature)[1]
    w = np.mat(np.zeros((n,1)))
    it = 0
    while it <= iterMax:
        g = first_derivativ(feature, label, w) # 一阶导数
        G = second_derivative(feature) #二阶导数

        d = -G.I * g
        m = get_min_m(feature, label, sigma, delta, d, w, g)#得到最小的m
        w = w + pow(sigma, m) * d
        if it % 10  == 0 :
            print("\t---- itration: " + str(it) + " , error: " + str(get_error(feature, label, w)[0, 0]))
        it += 1
    return w

def load_data(file_path):
    '''导入数据
    input:  file_path(string):训练数据
    output: feature(mat):特征
            label(mat):标签
    '''
    f = open(file_path)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close()
    return np.mat(feature), np.mat(label).T

def save_model(file_name, w):
    '''保存最终的模型
    input:  file_name(string):要保存的文件的名称
            w(mat):训练好的线性回归模型
    '''
    f_result = open(file_name, "w")
    m, n = np.shape(w)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(w[i, j]))
        f_result.write("\t".join(w_tmp) + "\n")
    f_result.close()


if __name__ == "__main__":
    # 1、导入数据集
    print("----------- 1.load data ----------")
    feature, label = load_data("train_data.txt")
    # 2.1、最小二乘求解
    print("----------- 2.training ----------")
    # print("\t ---------- least_square ----------")
    # w_ls = least_square(feature, label)
    # 2.2、牛顿法
    print("\t ---------- newton ----------")
    w_newton = newton(feature, label, 50, 0.1, 0.5)
    # 3、保存最终的结果
    print("----------- 3.save result ----------")
    save_model("weights", w_newton)
