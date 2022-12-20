# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 16:26
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : torch2onnx.py
# @Software: PyCharm
import onnxruntime
import torch
import numpy as np
model = None

output_name = 'MSBDN_RDFF.onnx'
batch_size = 1
fake_x = torch.rand(1, 3, 256, 256, requires_grad=False)

dynamic_params = None

torch.onnx.export(
    model,
    fake_x.cuda(),
    output_name,
    export_params=True,
    opset_version=13,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=dynamic_params
)
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_model = onnxruntime.InferenceSession('MSBDN_RDFF.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
data = np.random.rand(1, 3, 256, 256).astype(np.float32)
onnx_input = {onnx_model.get_inputs()[0].name: data}
outputs = onnx_model.run(None, onnx_input)

tensor_data = torch.tensor(data)
tensor_outputs = model(tensor_data)



