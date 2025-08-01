# 一、基础
device
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
nn.sequential

PyTorch 中用于构建顺序神经网络模型的容器类，它可以按顺序将多个神经网络层组合在一起，形成一个序列化的网络结构。输入数据会按照顺序依次通过这些层进行处理，简化了模型定义的代码。
```
model = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```
nn.Linear

nn.Relu

nn.Dropout

## 损失函数

CrossEntropyLoss

# 二、案例



