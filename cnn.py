import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#import cv2 as cv
import random
import csv
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers, datasets, models
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.utils import class_weight 

#获取图片，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir):
    '''
    image_list=[]
    label_list=[]
    for file in os.listdir(file_dir ):
        a = file_dir + '/' + file
        image_list.append(a)
        label = file[0]
        label_list.append(label)
    '''
    Bread = []
    label_Bread = []
    DairyProduct = []
    label_DairyProduct = []
    Dessert = []
    label_Dessert = []
    Egg = []
    label_Egg = []
    FriedFood = []
    label_FriedFood = []
    Meat = []
    label_Meat = []
    Noodles = []
    label_Noodles = []
    Rice = []
    label_Rice = []
    Seafood = []
    label_Seafood = []
    Soup = []
    label_Soup = []
    Vegetable = []
    label_Vegetable = []

    for file in os.listdir(file_dir + '/Bread'):
        Bread.append(file_dir + '/Bread' + '/' + file)
        label_Bread.append(0)

    for file in os.listdir(file_dir + '/DairyProduct'):
        DairyProduct.append(file_dir + '/DairyProduct' + '/' + file)
        label_DairyProduct.append(1)

    for file in os.listdir(file_dir + '/Dessert'):
        Dessert.append(file_dir + '/Dessert' + '/' + file)
        label_Dessert.append(2)

    for file in os.listdir(file_dir + '/Egg'):
        Egg.append(file_dir + '/Egg' + '/' + file)
        label_Egg.append(3)

    for file in os.listdir(file_dir + '/FriedFood'):
        FriedFood.append(file_dir + '/FriedFood' + '/' + file)
        label_FriedFood.append(4)

    for file in os.listdir(file_dir + '/Meat'):
        Meat.append(file_dir + '/Meat' + '/' + file)
        label_Meat.append(5)

    for file in os.listdir(file_dir + '/Noodles'):
        Noodles.append(file_dir + '/Noodles' + '/' + file)
        label_Noodles.append(6)

    for file in os.listdir(file_dir + '/Rice'):
        Rice.append(file_dir + '/Rice' + '/' + file)
        label_Rice.append(7)

    for file in os.listdir(file_dir + '/Seafood'):
        Seafood.append(file_dir + '/Seafood' + '/' + file)
        label_Seafood.append(8)

    for file in os.listdir(file_dir + '/Soup'):
        Soup.append(file_dir + '/Soup' + '/' + file)
        label_Soup.append(9)

    for file in os.listdir(file_dir + '/Vegetable'):
        Vegetable.append(file_dir + '/Vegetable' + '/' + file)
        label_Vegetable.append(10)
    
    # 合并数据
    image_list = np.hstack((Bread, DairyProduct, Dessert, Egg, FriedFood, Meat,Noodles,Rice,Seafood,Soup,Vegetable))
    label_list = np.hstack((label_Bread, label_DairyProduct, label_Dessert, label_Egg, label_FriedFood, label_Meat,label_Noodles,label_Rice,label_Seafood,label_Soup,label_Vegetable))
    #利用shuffle打乱数据

    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    #将所有的image和label转换成list
    image_list = list(temp[:, 0])
    image_list = [i for i in image_list]
    label_list = list(temp[:, 1])
    label_list = [int((i)) for i in label_list]

    return image_list, label_list

def get_files1(file_dir):
    test = []
    for file in os.listdir(file_dir):
        test.append(file_dir +  '/' + file)
    return test

def get_tensor(image_list, label_list=[]):
    ims = []
    for image in image_list:
	    #读取路径下的图片
	    x = tf.io.read_file(image)
	    #将路径映射为照片,3通道
	    x = tf.image.decode_jpeg(x, channels=3)
	    #修改图像大小
	    x = tf.image.resize(x,[32,32])
	    #将图像压入列表中
	    ims.append(x)
    #将列表转换成tensor类型
    img = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(label_list)
    return img,y

def plot_learning_curves(history,epoch):
    '''将tra_accuracy和val—_accyracy绘图'''
    plt.plot(np.arange(1,epoch+1),history.history['accuracy'],label='tra_accuracy')
    plt.plot(np.arange(1,epoch+1),history.history['val_accuracy'],label='val_accuracy') #x轴次数随epoch更改
    plt.grid(True) #显示网格
    #plt.gca().set_ylim(0,1) #设置y轴范围
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend() # 将样例显示出来
    plt.show()


def plot_confusion_matrix(cm, labels_name, title):
    '''绘制混淆矩阵'''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    # 显示数据
    for i in range(len(cm)):    #第几行
        for j in range(len(cm[i])):    #第几列
            plt.text(i, j, round(cm[i][j],2),horizontalalignment="center",color="white" if cm[i, j] > 0.2 else "black",fontsize=5)
    plt.show()

if __name__ == "__main__":

    labels_name = ['Bread', 'DairyProduct', 'Dessert', 'Egg', 'FriedFood', 'Meat','Noodles','Rice','Seafood','Soup','Vegetable']
    #训练图片的路径
    train_dir = 'training1'
    test_dir = 'testing'
    val_dir = 'validation1'
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    #训练图片与标签
    image_list, label_list = get_files(train_dir)
    train_x, train_y = get_tensor(image_list, label_list=label_list)
    
    #验证图片与标签
    val_image_list,val_label_list = get_files(val_dir)
    val_x, val_y = get_tensor(val_image_list,label_list=val_label_list)
    
    # 测试集
    test_list = get_files1(test_dir)
    test,_ = get_tensor(test_list)
   
    #归一化
    train_x, val_x = train_x / 255.0, val_x / 255.0
    print('train_x shape:', train_x.shape, 'val_x shape:', val_x.shape)
    

    # 模型构造
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))
    model.summary()

    # 编译
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    
    #训练集和验证集训练
    epoch = 30
    #class_weights = class_weight.compute_class_weight('balanced',np.unique(label_list),label_list)
    #class_weight_dict = dict(enumerate(class_weights))
    history = model.fit(train_x, train_y, epochs=epoch,validation_data=(val_x,val_y)) #epoch可更改
    model.evaluate(val_x,val_y)

    #绘制准确率图
    plot_learning_curves(history,epoch)
    
    #绘制混淆矩阵图
    y_pred = []
    y_pred1 = model.predict(val_x)
    y_pred2 = list(y_pred1)
    for x in y_pred2:
        y_pred.append(np.argmax(x))
    y_true = val_label_list
    cm = confusion_matrix(y_true,y_pred)
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
 
    


    #测试集预测
    predict = model.predict(test)
    pre = list(predict)

    x_max = []
    for x in pre:
        x_max.append(np.argmax(x))
    result = {'result':x_max}
    df = pd.DataFrame(result)
    df.to_csv("test_result.csv")
    


