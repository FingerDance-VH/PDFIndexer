import io
import os
from typing import List

import fitz
import numpy as np
import pdfplumber

import urllib
import urllib.parse

from tqdm import tqdm
from bs4 import BeautifulSoup
import re

import pandas as pd

from transformers import AutoTokenizer, AutoModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

import pickle

from jina import Document, DocumentArray, Executor, requests, Flow
from jina.logging.logger import JinaLogger


'''
如果使用预训练过的模型，就必须要匹配它在预训练时的最大长度（这里的模型在与训练时似乎最大长度都是512）
'''
# Roberta模型
roberta_model = AutoModel.from_pretrained("xlm-roberta-base")
#需要移动到cuda上
roberta_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# CLIP模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#需要移动到cuda上
clip_model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


class PDFIndexer(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(context=self.__class__.__name__)

    @requests
    def indexEncoder(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # 处理文本
            pdf_text = self.get_pdf_text(doc)
            pdf_text_list = self.split_text(pdf_text)
            text_tensor_list = self.textEncoder(pdf_text_list)
            print(len(text_tensor_list))
            d1 = Document(text='text_tensor_list')
            d1.chunks.extend([Document(tensor=text_tensor) for text_tensor in text_tensor_list])
            doc.chunks.extend([d1])

            # 处理图片
            pdf_img = self.get_pdf_img(doc)
            img_tensor_list = self.imgEncoder(pdf_img)
            print(len(img_tensor_list))
            d2 = Document(text='img_tensor_list')
            d2.chunks.extend([Document(tensor=img_tensor) for img_tensor in img_tensor_list])
            doc.chunks.extend([d2])

            # 获取PDF标题
            title = self.extractTitle(doc)
            print(title)
            doc.chunks.extend([Document(text=title)])

            # 获取PDF大小
            size = self.get_pdf_size(doc)
            print(size)
            doc.chunks.extend([Document(text=size)])

    # 获取PDF文本
    def get_pdf_text(self, doc: Document):
        try:
            pdf = fitz.open(doc.uri)
        except:
            print('只接受文件URI，不支持blob')
        # 将pdf转成html
        # pdf = fitz.open(file_path)
        html_content = ''
        for page in tqdm(pdf):
            html_content += page.get_text('html')

        html_content += "</body></html>"

        # 使用Beautifulsoup解析本地html
        soup = BeautifulSoup(html_content, "html.parser")
        com_text = []
        for div in soup.find_all('div'):
            full_text = []
            for p in div:
                text = str()
                for span in p:
                    p_info = '<span .*?>(.*?)</span>'  # 提取规则
                    res = re.findall(p_info, str(span))  # findall函数
                    if len(res) == 0:
                        pass
                    else:
                        text += res[0]  # 将列表中的字符串内容合并加到行字符串中
                full_text.append(text)
            com_text.append("".join(full_text))  # 合并
        com_text = "".join(com_text)  # 再次合并

        # 清洗文本
        string = re.sub(r"[\$\(\)\*\+\-\[\]\^\{\}\|#&_□■━◇◆○●�]", "", com_text)
        new_text = re.sub(r"  ", "", string)

        return new_text

    # 文本切分函数
    def split_text(self, content):
        # 最终文件的暂存字典列表
        dic_list = []

        # 判断文本属于中文还是英文
        isCH = False
        for ch in content:
            if u'\u4e00' <= ch <= u'\u9fff':
                isCH = True
                break

        try:
            # 拆分过长文本
            if not isCH:
                # 英文
                sent_list = re.findall(r'.{2200}', content)
                last = len(content) % 2200
            else:
                # 中文
                sent_list = re.findall(r'.{450}', content)
                last = len(content) % 450

            last_sent = content[-int(last):]
            if type(last_sent) == str:
                sent_list.append(last_sent)
        except:
            print(f'pdf文档文本切分失败')

        return sent_list

    # 获取PDF图片
    def get_pdf_img(self, doc: Document):
        try:
            pdf = pdfplumber.open(doc.uri)
        except:
            print('只接受文件URI，不支持blob')
        image_list = []
        for i in pdf.pages:
            images = i.images
            if images:
                for image in images:
                    image_list.append(image)
        # print(pdf.metadata)
        # print(len(image_list))
        return image_list

    # 文本编码器
    def textEncoder(self, text_list):
        # 获取每一段切分文本的编码
        roberta_output_list = []
        # xlm-roberta-base编码
        for i in text_list:
            encoded_input = tokenizer(i, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            for x in encoded_input:
                encoded_input[x] = encoded_input[x].to(device)
            with torch.no_grad():
                output = roberta_model(**encoded_input).last_hidden_state
            roberta_output_list.append(output)
        return roberta_output_list
        #     # 将张量编码为二进制
        #     roberta_output_list.append(output.cpu().detach().numpy().tobytes())
        # return pickle.dumps(roberta_output_list)

    # 图片编码器
    def imgEncoder(self, image_list):
        imgTensor_list = []
        image_features = []
        img_tensor_byte = []
        # 获取每一张图片的编码
        # Clip图片编码
        for index, img_data in enumerate(image_list):
            image_stream = io.BytesIO(img_data["stream"].get_data())
            try:
                img = Image.open(image_stream)
                imgTensor_list.append(img)
            except:
                print(f'第{index}张图片损坏')
                continue
            # img.close()
            # image_stream.close()
        if imgTensor_list == []:
            return img_tensor_byte
        else:
            # Clip图片编码
            inputs = processor(images=imgTensor_list, return_tensors="pt")
            for x in inputs:
                inputs[x] = inputs[x].to(device)
            # pixel_values = inputs['pixel_values'].to(device)
            # 获取编码结果
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            return image_features
            # 将张量编码为二进制
            # for i in image_features:
            #     img_tensor_byte.append(i.cpu().detach().numpy().tobytes())
            # # print(len(img_tensor_byte))
            # return pickle.dumps(img_tensor_byte)

    # 获取文件大小
    def get_pdf_size(self, doc: Document):
        try:
            file_path = doc.uri
        except:
            print('只接受文件URI，不支持blob')
        with open(file_path, 'rb') as f:
            # 使用io库将文件读取到内存中
            stream = io.BytesIO(f.read())
            # 获取文件大小（字节数）
            file_size = stream.getbuffer().nbytes
            KB = round(file_size / 1024, 1)
            if len(str(KB).split(sep='.')[0]) > 3:
                MB = round(KB / 1024, 1)
                size = str(MB) + 'MB'
            else:
                size = str(KB) + 'KB'
        return size

    # 提取标题
    def extractTitle(self, doc: Document):
        try:
            input_path = doc.uri
        except:
            print('只接受文件URI，不支持blob')

        file = input_path.split(sep='/')[-1][:-4]
        # 判断原文件名中是否含有URL编码
        pattern2 = re.compile('%[0-9a-fA-F]{2}')
        # 如果包含URL编码，则转码
        if bool(pattern2.search(file)):
            url_decoded = urllib.parse.unquote(file)
            url_decoded2 = ''.join(url_decoded.split(sep='/'))
            return url_decoded2

        try:
            pdf = fitz.open(input_path)
        except:
            print(input_path, 'pdf文件损坏')

        for page in tqdm(pdf):
            html_content = page.get_text('html')
            break

        # 使用Beautifulsoup解析本地html
        soup = BeautifulSoup(html_content, "html.parser")
        try:
            p_list = soup.find_all('p')
        except:
            print(input_path, '首页没有p标签')
        height_list = []
        max_height = 0
        p_info = '.*?line-height:(.*?)pt'
        for p in p_list:

            # print(p)

            txt = p['style']
            height = float(re.findall(p_info, str(txt))[0])
            height_list.append(height)
            if height > max_height:
                max_height = height

        # print(max_height)

        max_index_list = []
        for i, h in enumerate(height_list):
            if h == max_height:
                max_index_list.append(i)

        # print(max_index_list)

        title = ['']
        p_info2 = '<span .*?>(.*?)</span>'  # 提取规则
        for idx in max_index_list:
            for span in p_list[idx]:
                res = re.findall(p_info2, str(span))  # findall函数
                if len(res) == 0 or res[0] == title[-1]:
                    pass
                else:
                    if u'\u4e00' <= res[0][0] <= u'\u9fff':  # 如果字符串首位是中文
                        title.append(res[0])
                    else:
                        title.append('' + res[0])  # 如果首字母是英文则添加空格
        title = ''.join(title).strip()

        # 如果太长了，尝试换个策略再抓一次，再不行就算了
        p_info4 = '<b><span .*?>(.*?)</span></b>'  # 提取规则
        if len(title.encode()) > 200:
            title = ['']
            for idx in max_index_list:
                for span in p_list[idx]:
                    res = re.findall(p_info4, str(span))  # findall函数
                if len(res) == 0 or res[0] == title[-1]:
                    pass
                else:
                    if u'\u4e00' <= res[0][0] <= u'\u9fff':  # 如果字符串首位是中文
                        title.append(res[0])
                    else:
                        title.append('' + res[0])  # 如果首字母是英文则添加空格
            title = ''.join(title).strip()

        # print(title)

        pre_height = max_height
        loop = 1
        while title == '' or len(title.encode()) < 8 or len(title.encode()) > 200:
            next_height = 0
            for i in height_list:
                if i > next_height and i < pre_height:
                    next_height = i
            pre_height = next_height
            next_height_list = []
            for i, h in enumerate(height_list):
                if h == next_height:
                    next_height_list.append(i)
            title = ['']
            for idx in next_height_list:
                for span in p_list[idx]:
                    res = re.findall(p_info2, str(span))  # findall函数
                if len(res) == 0 or res[0] == title[-1]:
                    pass
                else:
                    if u'\u4e00' <= res[0][0] <= u'\u9fff':  # 如果字符串首位是中文
                        title.append(res[0])
                    else:
                        title.append('' + res[0])  # 如果首字母是英文则添加空格
            title = ''.join(title).strip()

            # 如果太长了，尝试换个策略再抓一次，再不行就算了
            if len(title.encode()) > 200:
                title = ['']
                for idx in next_height_list:
                    for span in p_list[idx]:
                        res = re.findall(p_info4, str(span))  # findall函数
                    if len(res) == 0 or res[0] == title[-1]:
                        pass
                    else:
                        if u'\u4e00' <= res[0][0] <= u'\u9fff':  # 如果字符串首位是中文
                            title.append(res[0])
                        else:
                            title.append('' + res[0])  # 如果首字母是英文则添加空格
                title = ''.join(title).strip()

            print(f'循环{loop}次: ', title)
            loop += 1
            if loop == 5:
                break

        # 标题字节数不能超过200字节，不能小于4字节
        if len(title.encode()) > 4 and len(title.encode()) < 200:
            # 去掉首尾空格
            text = title.strip()

            # 将中英文混合的单词分隔开来
            pattern1 = re.compile(r'([\u4e00-\u9fa5]+|[a-zA-Z0-9]+)')
            words = pattern1.findall(text)

            # 去掉其中的空格
            words = [w.strip() for w in words]

            # 如果含有英文，则在英文单词前加上空格
            pattern3 = re.compile('[a-zA-Z0-9]')
            for i in range(len(words)):
                if bool(pattern3.search(words[i])):
                    words[i] = ' ' + words[i]
            # 将分隔开的单词拼接起来
            new_title = ''.join(words)
            # print(new_title)
            new_title2 = ''.join(new_title.split(sep='/'))
            # print(new_title2)
            if new_title2 != '':
                title = new_title2.strip()
            else:
                title = file
        else:
            title = file

        return title

