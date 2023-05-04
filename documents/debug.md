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

## 解决

下载正确的`*.whl`文件并安装即可！

---

# IndexError

```cmd
File "D:\OneDrive - bupt.edu.cn\文档\BUPT\LERS\model\lers.py", line 183, into_dense_adj
  dense_adj[src][tgt] = 1
IndexError: index 131 is out of bounds for dimension 0 with size 128
```
