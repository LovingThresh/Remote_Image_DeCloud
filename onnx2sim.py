# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 16:53
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : onnx2sim.py
# @Software: PyCharm
import onnx
from onnxsim import simplify
from torch2trt import torch2trt, TRTModule

onnx_model = onnx.load('MSBDN_RDFF.onnx')
model_sim, check = simplify(onnx_model)
assert check, "Not Validation"
onnx.save(model_sim, 'MSBDN_RDFF_sim.onnx')

