#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-09-02 20:43
# @Author : wangjue
# @Site : 
# @File : main.py
# @Software: PyCharm

import base64
import os
from keras import models, backend
import tensorflow as tf

import hashlib
import os
import pathlib
import requests
import scipy.fftpack


PATH = 'imgs'


import sys

import cv2
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


graph = tf.compat.v1.get_default_graph()

PATH = lambda p: os.path.abspath(
    os.path.join(os.path.dirname(__file__), p)
)


TEXT_MODEL = ""
IMG_MODEL = ""


def get_text(img, offset=0):
    """
    获取图片中的文字
    :param img:
    :param offset:
    :return:
    """

    text = _get_text(img, offset)
    text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    text = text / 255.0
    h, w = text.shape
    text.shape = (1, h, w, 1)
    return text


def base64_to_image(base64_code):
    """
    将
    :param base64_code:
    :return:
    """
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


class Verify:
    def __init__(self):
        self.textModel = ""
        self.imgModel = ""
        self.loadImgModel()
        self.loadTextModel()

    def loadTextModel(self):
        if not self.textModel:
            self.textModel = models.load_model(PATH('../model.v2.0.h5'))
        else:
            print("无需加载模型model.v2.0.h5")

    def loadImgModel(self):
        if not self.imgModel:
            self.imgModel = models.load_model(PATH('../12306.image.model.h5'))

    def verify(self, fn):
        verify_titles = ['打字机', '调色板', '跑步机', '毛线', '老虎', '安全帽', '沙包', '盘子', '本子', '药片', '双面胶', '龙舟', '红酒', '拖把', '卷尺',
                         '海苔', '红豆', '黑板', '热水袋', '烛台', '钟表', '路灯', '沙拉', '海报', '公交卡', '樱桃', '创可贴', '牌坊', '苍蝇拍', '高压锅',
                         '电线', '网球拍', '海鸥', '风铃', '订书机', '冰箱', '话梅', '排风机', '锅铲', '绿豆', '航母', '电子秤', '红枣', '金字塔', '鞭炮',
                         '菠萝', '开瓶器', '电饭煲', '仪表盘', '棉棒', '篮球', '狮子', '蚂蚁', '蜡烛', '茶盅', '印章', '茶几', '啤酒', '档案袋', '挂钟', '刺绣',
                         '铃铛', '护腕', '手掌印', '锦旗', '文具盒', '辣椒酱', '耳塞', '中国结', '蜥蜴', '剪纸', '漏斗', '锣', '蒸笼', '珊瑚', '雨靴', '薯条',
                         '蜜蜂', '日历', '口哨']
        # 读取并预处理验证码
        img = base64_to_image(fn)
        text = get_text(img)
        imgs = np.array(list(_get_imgs(img)))
        imgs = preprocess_input(imgs)
        text_list = []
        # 识别文字
        self.loadTextModel()
        global graph
        with graph.as_default():
            label = self.textModel.predict(text)
        label = label.argmax()
        text = verify_titles[label]
        text_list.append(text)
        # 获取下一个词
        # 根据第一个词的长度来定位第二个词的位置
        if len(text) == 1:
            offset = 27
        elif len(text) == 2:
            offset = 47
        else:
            offset = 60
        text = get_text(img, offset=offset)
        if text.mean() < 0.95:
            with graph.as_default():
                label = self.textModel.predict(text)
            label = label.argmax()
            text = verify_titles[label]
            text_list.append(text)
        print("题目为{}".format(text_list))
        # 加载图片分类器
        self.loadImgModel()
        with graph.as_default():
            labels = self.imgModel.predict(imgs)
        labels = labels.argmax(axis=1)
        results = []
        for pos, label in enumerate(labels):
            l = verify_titles[label]
            print(pos + 1, l)
            if l in text_list:
                results.append(str(pos + 1))
        return results


def preprocess_input(x):
    x = x.astype('float32')
    # 我是用cv2来读取的图片，其已经是BGR格式了
    mean = [103.939, 116.779, 123.68]
    x -= mean
    return x


# def load_data():
#     # 这是统计学专家提供的训练集
#     data = np.load('captcha.npz')
#     train_x, train_y = data['images'], data['labels']
#     train_x = preprocess_input(train_x)
#     # 由于是统计得来的信息，所以在此给定可信度
#     sample_weight = train_y.max(axis=1) / np.sqrt(train_y.sum(axis=1))
#     sample_weight /= sample_weight.mean()
#     train_y = train_y.argmax(axis=1)
#
#     # 这是人工提供的验证集
#     data = np.load('captcha.test.npz')
#     test_x, test_y = data['images'], data['labels']
#     test_x = preprocess_input(test_x)
#     return (train_x, train_y, sample_weight), (test_x, test_y)


def learn():
    (train_x, train_y, sample_weight), (test_x, test_y) = load_data()
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True)
    train_generator = datagen.flow(train_x, train_y, sample_weight=sample_weight)
    base = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    for layer in base.layers[:-4]:
        layer.trainable = False
    model = models.Sequential([
        base,
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(80, activation='softmax')
    ])
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    reduce_lr = ReduceLROnPlateau(verbose=1)
    model.fit_generator(train_generator, epochs=400,
                        steps_per_epoch=100,
                        validation_data=(test_x[:800], test_y[:800]),
                        callbacks=[reduce_lr])
    result = model.evaluate(test_x, test_y)
    print(result)
    model.save('12306.image.model.h5', include_optimizer=False)


def predict(imgs):
    imgs = preprocess_input(imgs)
    model = models.load_model('12306.image.model.h5')
    labels = model.predict(imgs)
    return labels


def _predict(fn):
    imgs = cv2.imread(fn)
    imgs = cv2.resize(imgs, (67, 67))
    imgs.shape = (-1, 67, 67, 3)
    labels = predict(imgs)
    print(labels.max(axis=1))
    print(labels.argmax(axis=1))



def download_image():
    # 抓取验证码
    # 存放到指定path下
    # 文件名为图像的MD5
    url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand'
    r = requests.get(url)
    fn = hashlib.md5(r.content).hexdigest()
    with open(f'{PATH}/{fn}.jpg', 'wb') as fp:
        fp.write(r.content)


def download_images():
    pathlib.Path(PATH).mkdir(exist_ok=True)
    for idx in range(40000):
        download_image()
        print(idx)


def _get_text(img, offset=0):
    # 得到图像中的文本部分
    return img[3:22, 120 + offset:177 + offset]


def avhash(im):
    im = cv2.resize(im, (8, 8), interpolation=cv2.INTER_CUBIC)
    avg = im.mean()
    im = im > avg
    im = np.packbits(im)
    return im


def phash(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(scipy.fftpack.dct(im, axis=0), axis=1)
    im = im[:8, :8]
    med = np.median(im)
    im = im > med
    im = np.packbits(im)
    return im


def _get_imgs(img):
    interval = 5
    length = 67
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]


def get_imgs(img):
    imgs = []
    for img in _get_imgs(img):
        imgs.append(phash(img))
    return imgs


def pretreat():
    if not os.path.isdir(PATH):
        download_images()
    texts, imgs = [], []
    for img in os.listdir(PATH):
        img = os.path.join(PATH, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        texts.append(get_text(img))
        imgs.append(get_imgs(img))
    return texts, imgs


def load_data(path='data.npz'):
    if not os.path.isfile(path):
        texts, imgs = pretreat()
        np.savez(path, texts=texts, images=imgs)
    f = np.load(path)
    return f['texts'], f['images']

if __name__ == '__main__':
    #pass
    # verify("verify-img1.jpeg")
    # f = open('./images/timg.jpg', 'wb')
    # print(f)
    f= cv2.imread('./images/timg.jpeg')

    text = get_text(f)
    print(text)
