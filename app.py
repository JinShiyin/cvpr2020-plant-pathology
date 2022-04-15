#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: __init__.py.py
@time: 2022/2/13 16:11
@version 1.0
@descwerkzeug:
"""
import argparse
from flask import Flask
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from api import blueprint as api
import service
import threading



def create_app():
    flask = Flask(__name__,
                  static_url_path='',
                  static_folder='dist')
    # flask.config['SECRET_KEY'] = 'secret!'
    # mount all blueprints from api module.
    flask.wsgi_app = ProxyFix(flask.wsgi_app)
    flask.register_blueprint(api)
    CORS(flask)
    return flask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='10.8.0.94', help='ip address of flask server in local network.')
    parser.add_argument('--port', type=int, default=5000, help='listening port of flask server in local network.')
    # parser.add_argument('--host', type=str, default='0.0.0.0', help='ip address of flask server in local network.')
    # parser.add_argument('--host', type=str, default='192.168.0.28', help='ip address of flask server in local network.')
    # parser.add_argument('--port', type=int, default=9527, help='listening port of flask server in local network.')
    parser.add_argument('--debug', type=bool, default=False, help='listening port of flask server in local network.')

    args = parser.parse_args()
    app = create_app()
    model_thread = threading.Thread(target=service.run)
    model_thread.start()
    app.run(host=args.host, port=args.port)
