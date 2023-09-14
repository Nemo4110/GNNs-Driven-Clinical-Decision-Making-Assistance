# 被拒稿2次后...

以下是编辑回复邮件中，关于拒稿理由的部分：

- In this case, we feel that the insights provided by the new data **do not offer a sufficient translational or clinical advance that would appeal to the broad readership of Nature Medicine.**
- In this case, we have no doubt that your findings on an objective model will be of interest to experts in laboratory items, EMRs, and biomedical informatics. However, I regret that we are **unable to conclude that the paper provides the sort of substantial practical or conceptual advance that would be of immediate interest to a broad readership of researchers in artificial intelligence or machine learning**. We feel that the present manuscript would be better suited to another journal than Nature Machine Intelligence.

关于被拒原因的推测：

- 这篇文章没有新意/创新点，太过于小众不适合期刊吸流量？
- 写作方面的问题，编辑一般花费数分钟快速阅读摘要、引言等部分，目前的故事无法引起编辑的兴趣/故事没讲清楚
- 期刊方面的问题，比如大量投稿挤压之类（可能性太小）

---

## 这篇工作做了什么

为了使用患者的历史检验结果去推荐检验项目，从序列推荐系统领域寻找可行方法，并根据我们的数据特点，对方法进行了改进。

### 与类似的工作相比，有什么不同/改进了哪些

- 与[借鉴的方法](https://arxiv.org/abs/2104.07368)对比：
  - 将动态图按`timestep`切分，单独对每一个切分子图使用`GNN`交互结点特征
  - 将时序特征交互部分的模型从`LSTM`、`GRU`等改进为`Transformer Decoder`
  - （杂思）若要证明上述改进有效，或许应该在同样任务、同样数据集的情况下进行实验，与原方法对比性能；
- 与序列推荐系统领域的其他方法对比：
  - 使用`GNN`聚合图中邻居节点的语义信息——近年的工作仍有仅仅使用用户/物品的`Embedding`或`Features`；
    - [Search-based Time-aware Recommendation with Sequential Behavior Data](https://dl.acm.org/doi/10.1145/3485447.3512117)
    - [Deep Interest Highlight Network for Click-Through Rate Prediction in Trigger-Induced Recommendation](https://dl.acm.org/doi/abs/10.1145/3485447.3511970)
  - 对用户、物品节点在图中随时间变化的时序特征进行建模

### 创新点是什么

- “新方法”（动态图学习）用于“新任务”（检验项目推荐）
- 对“新方法”的进一步改进：
  - （主要）将动态图进一步按`timestep`切分，单独对每一个切分子图使用`GNN`交互结点特征；
  - （次要）将时序特征交互部分的模型从`LSTM`、`GRU`等改进为`Transformer Decoder`；

### 关于创新点的思考

:confused:以*WWW'23*的这篇[同数据集的药物推荐工作](https://dl.acm.org/doi/10.1145/3485447.3511936)为例：

- "In this paper, we propose the Conditional Generation Net (COGNet) which introduces **a novel copy-or-predict mechanism** to generate the set of medicines."
- 其余部分的[2019年的一篇工作](https://ojs.aaai.org/index.php/AAAI/article/view/3905/3783)相差无几：
  - 使用了药物相互作用图`DDI`
  - 结合`EHR`与`DDI`
  - 编码患者、药物、诊断的历史信息
  - ···

## 在讲故事方面是否出现了问题

:pensive:参阅多个文章、站在编辑的角度阅读，感觉在讲故事方面确实出现了问题：

- 过于冗长的引言部分
  - 没有让读者在几分钟内明白本文的主旨：
    - 到底是使用动态图学习这一“新方法”实现检验项目推荐这一任务？
    - 还是改进了动态图学习这一“新方法”？
  - 2张用于说明的图片也无助于明确一个唯一主旨
  - 参阅的那些文章在引言部分均无上述问题，阅读下来，能够很明确的了解到：
    - 作者到底想解决的问题；
    - 之前的解决方法有什么不足；
    - 作者的方法做了哪些改进；
    - ···
- 摘要、标题
  - 解决引言部分的故事问题后，相信这部分能够自然地迎刃而解；

---

## 下一步计划

- 修改引言、摘要、标题（对讲的故事进行大修）；
- 关于进一步投稿的选项：
  - 重新向*NAT MACH INTELL*提交手稿（大概率还是拒）；
  - 套到IEEE格式的overleaf模板，向*WWW'24*投稿；
- 实验方面是否可以继续完善？
  - 增加`LSTM`、`GRU`等变体的实验结果？
  - 用几种推荐系统领域里的、不同的、常见的方法进行对比实验
- 转向推荐系统领域的任务，进行实验，看看本篇工作提出的方法是否在性能表现上超过目前的SOTA；
  - 商品推荐
  - 点击率预测
    - 上述任务在天池、AMAZON网站上均有公开可用的数据集；
