#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: categories.py
@time: 2022/2/14 15:28
@version 1.0
@desc:
"""
from time import time
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from service import *
from PIL import Image
import cv2
import io
import numpy as np

api = Namespace('photos', description='Photo Transmition')

image_upload = reqparse.RequestParser()
image_upload.add_argument('file', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG photos')

@api.route('/')
class Photos(Resource):

    @api.expect(image_upload)
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args['file']

        # image_path = '/data/lxd/datasets/2021-12-12-Eggs/Weak/174409520_Egg3_(ruopei--ok)_L_0_cam3.bmp'
        # image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        # image = image.transpose(1, 0, 2)

        pil_image = Image.open(io.BytesIO(image.read()))
        image = pil_image.convert('RGB')
        image = np.array(image)
        image = image.transpose(1, 0, 2)
        # pil_image.save(path)

        submission = []
        start = time()
        for idx in range(len(input_entry)):
            input_entry[idx].put(image)
        
        for idx in range(len(output_entry)):
            res = output_entry[idx].get()
            submission.append(res)
        end = time()

        print(f'Inference time {end - start: .2f}')
        return assemble_result(submission)

