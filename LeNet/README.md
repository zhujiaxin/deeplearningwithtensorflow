分别有tf1.x和tf2的版本

模型名称LeNet

论文地址:http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

TF2主要是keras api来实现模型

TF1.X主要是用基础的api来实现模型

目前在tf1.X中 预测得到的训练集精度与测试集精度均为98%附近，在restore model from ckpt中有记录，还有从CKPT读取参数的代码示例。

在tf2中，在验证集上精度要稍微低一些，为96% 

