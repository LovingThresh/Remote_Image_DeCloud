# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 20:08
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : trt_run.py
# @Software: PyCharm
# 导入必用依赖
import torch
import numpy as np
import tensorrt as trt
from torch2trt import TRTModule

# 加载日志记录器
logger = trt.Logger(trt.Logger.INFO)
# 加载engine
with open('MSBDN_RDFF_sim_engine.trt', 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

net = TRTModule(engine, input_names=['input'], output_names=['output'])

# 推理
data = np.random.rand(1, 3, 256, 256).astype(np.float32)
data_tensor = torch.from_numpy(data)
data_tensor = data_tensor.to('cuda')

# 结果
result = net(data_tensor)


