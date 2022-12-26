# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 9:22
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : trt_build.py
# @Software: PyCharm
import tensorrt as trt
onnx_path = 'MSBDN_RDFF_sim.onnx'
engine_path = 'MSBDN_RDFF.trt'
# 创建logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)

# 创建构建器builder
builder = trt.Builder(logger)
# 预创建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# 加载onnx解析器
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file(onnx_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))
if not success:
    pass  # Error handling code here
# builder配置
config = builder.create_builder_config()
# 分配显存作为工作区间，一般建议为显存一半的大小
config.max_workspace_size = 4 << 30  # 1 Mi
serialized_engine = builder.build_serialized_network(network, config)
# 序列化生成engine文件
with open(engine_path, "wb") as f:
    f.write(serialized_engine)
    print("generate file success!")
