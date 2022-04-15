#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: __init__.py.py
@time: 2022/2/13 19:39
@version 1.0
@descwerkzeug:
"""


from tkinter.tix import Tree
import requests
import os
import time
import cv2
import numpy as np
from scipy.special import softmax
from sklearn import metrics
import pandas as pd

def parse_labels(filenames):
    return [filename[:-4].split('_')[-4][1:-1] for filename in filenames]

def parse_label(filename):
   return filename[:-4].split('_')[-4][1:-1]

def render_label(image, label, pred):
    image = cv2.putText(image, f'Label: {label}', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Pred: {pred}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Classification Result',image)
    cv2.waitKey(1)
    return image

def save_csv_report(report, output_dir, class_names):
    repdf = pd.DataFrame(report).transpose()
    repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    repdf.to_csv(output_dir, index=False,float_format='%.4f')

def save_csv_confusion_matrix(confusion_matrix, output_dir, class_names):
    cm = pd.DataFrame(confusion_matrix, columns=class_names)
    cm.insert(loc=0, column=' ', value=class_names)
    cm.to_csv(output_dir, index=False)


# API url
api = 'http://10.8.0.94:5000/api/photos/'

class_label_to_name = {0: 'ok', 1: 'qishihuangdong', 2 : 'sipei', 3: 'kongliao', 4: 'wuqishi', 5: 'liewen', 6: 'ruopei'}
name_to_class_label = {'ok': 0 , 'qishihuangdong' : 1, 'sipei' : 1, 'kongliao' : 1, 'wuqishi' : 1, 'liewen' : 1, 'ruopei' : 1}
# name_to_class_label = {'ok': 0 , 'qishihuangdong' : 1, 'sipei' : 2, 'kongliao' : 3, 'wuqishi' : 4, 'liewen' : 5, 'ruopei' : 6}

test_file_path= '/Users/luvletteru/Documents/2022-02-13-Eggs-Test'
output_dir = '/Users/luvletteru/Documents/2022-02-16-Eggs-Test-Result'
result_dir = os.path.join(output_dir, 'results')
no_ok_dir = os.path.join(output_dir, 'no_ok')
bn_class_names = ['OK', 'NoOK']
# bn_class_names = ['好蛋', '气室抖动', '死胚', '空', '无气室', '裂纹', '弱胚']

os.makedirs(result_dir, exist_ok=True)
os.makedirs(no_ok_dir, exist_ok=True)
filenames = os.listdir(test_file_path)

num_samples = len(filenames)
# Parse labels from filenames
# label_names = parse_labels(filenames)
# labels = [name_to_class_label[label_name] for label_name in label_names if label_name in name_to_class_label]
label_names = []
labels = []

pred_label_names = []
pred_labels = []

start = time.time()

for idx, filename in enumerate(filenames):
    if '--' in filename:
        print(f"Filter {filename}")
        continue
    s = time.time()
    image_path = os.path.join(test_file_path, filename)
    image = cv2.imread(image_path)
    # Send classification request to inference server
    response = requests.post(api, files = {"file": open(image_path, 'rb')})

    # Obtain prediction result
    pred_label_name = response.json()

    # Ground Truth
    label_name = parse_label(filename)
    label = name_to_class_label[label_name]
    label_names.append(label_name)
    labels.append(label)
    # label_name = label_names[idx]
    # label = labels[idx]

    pred_label_names.append(pred_label_name)
    pred_label = name_to_class_label[pred_label_name]
    pred_labels.append(pred_label)
    # Save classification result
    image = render_label(image, label_name, pred_label_name)
    cv2.imwrite(os.path.join(result_dir, filename), image)
    if pred_label != label:
        cv2.imwrite(os.path.join(no_ok_dir, filename), image)
    # Estimate performance
    print(f'File: {filename}, Label: {label_name}, Pred: {pred_label_name}')
    e = time.time()
    print(f'Processing time {round(e - s, 2)} (s)')


pred_labels = np.array(pred_labels)
labels = np.array(labels)
bn_confusion_matrix =metrics.confusion_matrix(labels, pred_labels)
bn_report = metrics.classification_report(labels, pred_labels, target_names=bn_class_names, output_dict=True)
save_csv_report(bn_report, os.path.join(output_dir,f'BN_Report.csv'), bn_class_names)
save_csv_confusion_matrix(bn_confusion_matrix, os.path.join(output_dir,f'BN_CM.csv'), bn_class_names)
print(bn_confusion_matrix)
print(bn_report)

end = time.time()
total = end - start
print(f'{round(total / num_samples, 2)} (s), {round(num_samples / total, 2)} (fps)')


