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

下载正确的`*.whl`文件并安装即可！（后面发现不用安装这3个包也行）

---

# IndexError

## 报错信息

```shell
File "D:\OneDrive - bupt.edu.cn\文档\BUPT\LERS\model\lers.py", line 183, into_dense_adj
  dense_adj[src][tgt] = 1
IndexError: index 131 is out of bounds for dimension 0 with size 128
```

## 解决

Just because I mistook the shape arg:

- Wrong: `shape=(scores.shape[0], scores.shape[1]))`
- Right: `shape=(scores.shape[-2], scores.shape[-1]))`

As the scores tensor has shape of `[20, 128, 753]`

---

# 服务器重新登录校园网网关

## 问题描述

5.5这天发现服务器无法下载`d2l`包，并且无法`ping`通外网。

## 解决

找王智刚看了下，说这是因为服务器没连上北邮校内网关，重新登录网关就好。

```shell
curl 'http://10.3.8.211/login' --data 'user=<学号>&pass=<密码>'
```

---

# 模型训练过程中出现`NaN`

## 排查

:sob:起先两周的排查如下：

### :date: 数据相关排查

#### `subgraph`过程中出现`NaN`？

模型从数据集类中获取一份大的异质图后，首先将它按设定的最大时间步`max_timestep`拆分成一个个小的子图，每个子图中包含每一个时间步`max_timestep`下的图信息（相当于将动态图拆分为一个个静态图）：

```python
def forward(self, hg: HeteroData):
    hgs = [self.get_subgraph_by_timestep(hg, timestep=t) for t in range(self.max_timestep)]
    ······
```

为排查`NaN`是否出现在这一过程中，在`get_subgraph_by_timestep`中加入以下代码：

```python
······
# For debug NaN
assert not sub_hg["admission"].node_id.isnan().any()
assert not sub_hg["admission"].x.isnan().any()
assert not sub_hg["labitem"].node_id.isnan().any()
assert not sub_hg["labitem"].x.isnan().any()
assert not sub_hg["admission", "did", "labitem"].edge_index.isnan().any()
assert sub_hg["admission", "did", "labitem"].edge_index.shape[-1] > 0
assert not sub_hg["admission", "did", "labitem"].x.isnan().any()
assert not sub_hg['labitem', 'rev_did', 'admission'].x.isnan().any()
assert not sub_hg.labels.isnan().any()

return sub_hg
```

后面的测试中，没有一次出现上面代码行的`assert`报错，因此排除这里的可能。

#### 数据预处理的问题？

首先排查了从处理好的`.csv`文件按不同的`batch_size`生成异质图文件这一过程中存在`NaN`的可能，重新设定`batch_size`=256, 512生成了包含不同大小`hadm`结点的异质图，并保存：

```python
# TODO: try 64? 256? 512? 1024? ...
batch_size = 512
list_df_admissions_single_batch_train, list_df_labevents_single_batch_train = batches_spliter(list_train_hadmid, df_admissions, df_labevents, batch_size=batch_size)
list_df_admissions_single_batch_val,   list_df_labevents_single_batch_val   = batches_spliter(list_val_hadmid,   df_admissions, df_labevents, batch_size=batch_size)
list_df_admissions_single_batch_test,  list_df_labevents_single_batch_test  = batches_spliter(list_test_hadmid,  df_admissions, df_labevents, batch_size=batch_size)

......

train_hgs = [construct_dynamic_hetero_graph(df_admissions_single_batch, df_labitems, df_labevents_single_batch) for df_admissions_single_batch, df_labevents_single_batch in tqdm(zip(list_df_admissions_single_batch_train, list_df_labevents_single_batch_train))]
val_hgs   = [construct_dynamic_hetero_graph(df_admissions_single_batch, df_labitems, df_labevents_single_batch) for df_admissions_single_batch, df_labevents_single_batch in tqdm(zip(list_df_admissions_single_batch_val,   list_df_labevents_single_batch_val))]
test_hgs  = [construct_dynamic_hetero_graph(df_admissions_single_batch, df_labitems, df_labevents_single_batch) for df_admissions_single_batch, df_labevents_single_batch in tqdm(zip(list_df_admissions_single_batch_test,  list_df_labevents_single_batch_test))]

......saving .pt file

```

后续的测试中，还是一样地发生模型输出`NaN`数据的问题（`get_subgraph_by_timestep`中的`assert`仍然没有触发，说明数据在输入到模型前都是没有问题的）！

但是，翻阅[PyTorch论坛中的讨论](https://discuss.pytorch.org/t/gradient-becomes-nan-with-random-runtimeerror-function-expbackward0-returned-nan-values-in-its-0th-output/166900)，看到了以下回复：

>Non nan losses and nan gradients are mostly a result of some absurd (undefined) mathematical operation like 0⁰, dividing by 0 and so on.
>
> The hint provided by anomaly detection probably hints at the step in the computational graph where such an operation is occurring leading to nan gradients.
>
> I would also think limited precision can be a potential reason but less likely as compared to the former.

于是，想到了我在数据预处理中，用0进行了`fillna`；为排除这种做法导致模型输出`NaN`数据的可能，回头修改`fillna`的填充值为$1e-11$，然后重新生成了异质图文件。

后续测试中，模型仍然输出`NaN`，所以排除不同`fillna`值导致的可能。

### :bar_chart: 超参数设置排查

- 修改学习率：改大改小都仍然有`NaN`输出
- 将模型参数、输入数据的精读改为`torch.float64`：训练速度变慢、模型大小变大，但仍然有`NaN`输出
- 更改模型层数/维度：仍然有`NaN`输出
  - `nn.TransformerDecoder`中`num_layers`=3/6
  - `nn.TransformerDecoderLayer`中`dim_feedforward`=2048/512
- 使用GPU/使用CPU训练：均有`NaN`输出
- 使用梯度裁剪`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)`：仍然`NaN`

### :file_folder: 环境/包版本问题排查

- 重启服务器（遇到了开关机问题、网关登录问题，好在均解决了）
- 重装`conda`环境
- 更换`PyG`、`PyTorch`版本（在[『PyTorch的某个Github讨论贴』](https://github.com/pytorch/pytorch/issues/45724)中，发现作者与提问者在不同版本下，一个没有`NaN`问题，另一个有；于是，尝试看看更换版本能不能解决问题）

后续测试中，均有`NaN`输出，可以排除环境/包版本导致的可能。

### :notebook: 模型本身问题排查

#### 某个GNN模型导致的问题？

最初写好的模型中，提取异质图中特征信息的GNN模型使用的是经过`to_hetero`转换后的`GINEConv`模型。为了排除单一模型代码有误导致错误的可能，更换了以下GNN模型进行测试：

- GENConv
- GATConv

均和`GINEConv`模型一样，训练到某一时刻有`NaN`输出。

#### 激活函数导致的问题？

在PyTorch论坛找到一个贴子[『Relu function results in nans』](https://discuss.pytorch.org/t/relu-function-results-in-nans/122019)，上面说激活函数有时会导致`NaN`。于是，将模型中自己加上激活函数的地方进行了以下测试：

- 取消激活函数
- 更改激活函数（比如改为`ELU`）

```python
# In SingleGnn
node_feats = self.conv1(x=node_feats, edge_index=edge_index,edge_attr=edge_attrs).relu()
node_feats = self.conv2(x=node_feats, edge_index=edge_index,edge_attr=edge_attrs).relu()
...
# In LERS
scores = torch.matmul(admission_node_feats, labitem_node_feats.transpose(1, 2)).sigmoid()
```

仍然产生`NaN`。

#### `nn.TransformerDecoder`前向过程中`mask`的`-inf`遮蔽值导致的问题？

<center>
    <img src="assets/mask_qk.jpg" style="zoom: 69%;"><br>
</center>

```python
tgt_mask = memory_mask = nn.Transformer.generate_square_subsequent_mas(self.max_timestep).to(device)
# tgt_mask = memory_mask = torch.triu(torch.full((self.max_timestep, self.max_timestep), -1e+31, device=device), diagonal=1)
```

将遮蔽值改成$-1e31$试了下，仍然出现`NaN`数据，于是排除这一可能。

#### 损失函数问题？

一开始仅尝试更换了损失函数：

```python
loss_func = torch.nn.L1Loss()
# loss_func = torch.nn.BCELoss()
# loss_func = torch.nn.KLDivLoss(reduction="batchmean")
```

测试结果都仍有`NaN`数据产生。

然而，开启`torch.autograd.detect_anomaly`后，发现每次最初发现`NaN`异常都是在`loss.backward()`这里：

```bash
Traceback (most recent call last):
  File "/data/data2/041/LERS/main.py", line 78, in <module>
    loss.backward()
  File "/data/data2/041/envs/LERS/lib/python3.9/site-packages/torch/_tensor.py", line 396, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/data/data2/041/envs/LERS/lib/python3.9/site-packages/torch/autograd/__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: Function 'SigmoidBackward0' returned nan values in its 0th output.
```

更进一步地，在本地机子上使用以下代码，仅前向无反向地进行了测试：

```python
if __name__ == "__main__":
    train_set = MyOwnDataset(root_path=r"D:\Datasets\mimic\mimic-iii-hgs\batch_size_128", usage="train")
    model = LERS(max_timestep=20, gnn_type="GINEConv")

    for hg in train_set:
        scores, labels = model(train_set[0])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # print(scores)  # torch.Size([20, 128, 753])
        print(scores.isnan().any(), loss.isnan().any())
```

其输出均为`False, False`！~~说明**若不进行梯度的反向传播，模型是不会有`NaN`产生的！**~~（侧面反映了模型结构应该是没有问题的）

## 解决！

:grin:联系了郝师兄，请他帮忙分析代码，找找问题出在哪儿。

### 问题出在哪儿？

> 调用官方写好的模型，出问题的可能性很小

于是我们将排查重点放在数据上，进而发现：

- 模型在每次都是在迭代**固定顺序**的数据集文件到某一个特定的文件时发生`NaN`”
- 数据的打乱（或称之为`shuffle`）发生在生成`.pt`数据集文件之前（见`construct_grapg.ipynb`），并且打乱的是病案号`HADM_ID`；也就是说，每次调用`MyOwnDataset`迭代遍历数据集时，顺序是固定的
- 尝试删除特定的文件（此时假设仅一个文件存在"**坏点**"，删除它就能解决问题），还是有`NaN`产生，说明不止一个数据文件有问题

因此，往前倒查，检查数据预处理中是否存在问题。终于，使用以下代码，在`LABEVENTS_NEW_remove_duplicate_edges.csv.gz`中，发现了问题所在：

```python
print(f"min: {df_labevents['CATAGORY'].min()}, max: {df_labevents['CATAGORY'].max()}")
print(f"min: {df_labevents['VALUENUM_Z-SCORED'].min()}, max: {df_labevents['VALUENUM_Z-SCORED'].max()}")
>>> min: 0.0, max: 1354.0
>>> min: -192.7052383450666, max: inf
```

`CATAGORY`，`VALUENUM_Z-SCORED`最终均作为边特征`edge_attr`$\in R^2$，因此`VALUENUM_Z-SCORED`的值中存在`inf`这样的"**坏点**"，自然会导致模型在训练中出现`NaN`。

于是，继续往前倒查生成`VALUENUM_Z-SCORED`的代码，终于发现问题所在！

```python
···
dfx['VALUENUM_Z-SCORED'] = dfx['VALUENUM'].apply(lambda x: (x-mean) / std)
···
```

当`std`为0或者是非常非常接近0的数值时，就会发生溢出，导致`VALUENUM_Z-SCORED`的值为`inf`！

### 如何解决

解决方法也非常简单:joy:，只需==给除数加上一个非常小的数==，如$1e-17$。随后重新生成数据集文件，训练和验证过程得以顺利完成。
