#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-10-09 19:04
# @Author : wangjue
# @Site : 
# @File : 爬取12306验证码.py
# @Software: PyCharm


import requests
import time
from PIL import Image
import matplotlib.pyplot as plt

class Login(object):
    def __init__(self):
        # 实例化session,自动携带cookie
        self.session = requests.session()
        # headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36'
        }
        # 获取登录页面
        self.login_url = 'https://kyfw.12306.cn/otn/login/init'
        # 验证码下载链接
        self.load_image_url = 'https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login'
        # 'https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login&rand=sjrand&0.8715477478180387'
        # 验证码图片链接
        self.captcha_check_url = 'https://kyfw.12306.cn/passport/captcha/captcha-check'
        # 密码登录
        self.user_login_url = 'https://kyfw.12306.cn/passport/web/login'

    def login(self):  # 登录界面
        # 暂时不需要登录页面信息
        pass

    def image_code_number(self,num):
        option = {
            '1': '40,40',
            '2': '110,40',
            '3': '180,40',
            '4': '260,40',
            '5': '40,110',
            '6': '100,110',
            '7': '180,40',
            '8': '260,40'
        }

        # 判断num长度
        # 点击验证码图像数量不一定，如果是1需要特殊处理
        check_num = []
        if len(num) == 1:
            check_num = option[num]
        else:
            image_num = num.split(',')  # num: '1,2'

            for i in image_num:
                check_num.append(option[i])
            check_num = ','.join(check_num)
        print(check_num)  # 40,40,110,40
        return check_num

    def load_verify_image(self):  # 下载图片验证图片
        '''
            answer: 120,40
            login_site: E
            rand: sjrand
        '''
        response = self.session.get(url=self.load_image_url,headers=self.headers)
        contents = response.content
        # 保存图片
        runtime = time.time()
        with open('check_image%s.jpg'%runtime, 'wb') as f:
            f.write(contents)
        print('图片下载成功!')

    def show_image(self):
        # img = Image.open('check_image.jpg')
        # img.show()

        # img=Image.open('F:/heck_image.jpg')
        img = Image.open('check_image.jpg')
        # 设置多个figure,设置figure的标题
        plt.figure("check_image")
        plt.imshow(img)
        plt.show()

    def verify_image_code(self):
        num = input('请输入正确的验证码：')
        # 构建formdata
        data = {
            'answer': self.image_code_number(num),
            'login_site': 'E',
            'rand': 'sjrand'
        }
        response = self.session.post(self.captcha_check_url,data=data,headers=self.headers)

        print(response.text)

    def user_login(self):
        '''
            username: 133
            password: 12313123
            appid: otn
        '''
        data = {
            'username': '***',
            'password': '***',
            'appid': 'otn'
        }
        response = self.session.post(self.user_login_url, data=data, headers=self.headers)
        print(response.text)

    def run(self):  # 主要逻辑实现
        # 下载图片验证码
        self.load_verify_image()
        # 显示下载的图片
        self.show_image()
        # 验证图片验证码
        self.verify_image_code()
        # 登录
        self.user_login()


railway = Login()
railway.run()