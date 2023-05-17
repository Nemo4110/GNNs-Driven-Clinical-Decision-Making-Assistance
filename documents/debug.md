# `ImportError: 'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'`

## 原因

没有安装以下三个包导致：

- `torch-scatter`
- `torch-sparse`
- `pyg-lib`

## 注意

- 安装时选择的`.whl`文件需要与python版本、pytorch版本、CUDA版本**相对应**！！！
- 若使用`conda`安装了错误版本，pytorch会退化为cpu版本！此时只能删除环境全部重来！
- `.whl`网址：<https://pytorch-geometric.com/whl/torch-1.12.1%2Bcu113.html>

## :wrench:解决

下载正确的`*.whl`文件并安装即可！

---

# IndexError

## 报错信息

```shell
File "D:\OneDrive - bupt.edu.cn\文档\BUPT\LERS\model\lers.py", line 183, into_dense_adj
  dense_adj[src][tgt] = 1
IndexError: index 131 is out of bounds for dimension 0 with size 128
```

## :wrench:解决

Just because I mistook the shape arg:

- Wrong: `shape=(scores.shape[0], scores.shape[1]))`
- Right: `shape=(scores.shape[-2], scores.shape[-1]))`

As the scores tensor has shape of `[20, 128, 753]`

---

# 服务器重新登录校园网网关

## 问题描述

5.5这天发现服务器无法下载`d2l`包，并且无法`ping`通外网。

## :wrench:解决

找王智刚看了下，说这是因为服务器没连上北邮校内网关，重新登录网关就好。

```shell
curl 'http://10.3.8.211/login' --data 'user=<学号>&pass=<密码>'
```

---

# 模型输出中出现`NaN`数据

## 排查

定位到`NaN`值是在训练过程中某一次`gnns`更新时产生的。（后面排除了前向过程出现问题的可能）

~~每次都在迭代到第60个数据集文件时发现`NaN`数据~~

`mask`中`-inf`与前面层的负输出相加，导致`NaN`？~~尝试在前面加`relu`看看？~~

在本地上仅`forward`无`backward`测试了完整数据集，无`NaN`出现。==说明不是前向过程出问题！==

## 尝试

- ~~更换`torch_geometric.nn.conv`~~
- ~~更换`loss_func`~~
- ~~重新生成不同`batch_size`的数据集~~
- ~~更换`PyTorch`版本~~
- ~~更改`labevent`中`fillna`的填充值~~
- ~~`mask`问题？~~
