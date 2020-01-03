# HAN

[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

### 模型实现

参考：[Github: sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification)

在此基础上进行了以下修改：

- 训练过程中在测试集上做评价，记录最佳结果
- 实现了多次训练求平均结果，相当于用不同的种子进行数据打乱训练
- 使用tensorboard进行loss、acc 等指标可视化

### 实验1：初始化方式

探讨不同的词向量初始化方式对实验结果的影响

#### 模型设置

- 从训练集中剔除词频小于5的词，构建词表(158774)。
- 使用Yahoo answers数据集，训练集1400000，batch size 512，epoch 2，共2735x2步
- 模型在训练时每100步进行评价，记录测试集上最佳精确率，五次结果取平均

#### 实验结果

- Rand1：随机初始化200维词向量，nn.Embedding 默认设置，内部使用nn.init.normal_，即(0,1)正态分布。
- Rand2：随机初始化200维词向量，nn.init.uniform_保证均匀分布，边界为 np.sqrt{3/200}
- Glove：使用glove.840B.300d.w2v作为预训练词向量，包含词表158774中的词汇129323(0.8145)个。
- Gensim：使用gensim工具在训练集上训练词向量，skipgram算法，200维，窗口10，最小词频5

|      | Rand1(epoch2) | Rand1(epoch4) | Rand1(epoch8) | Rand2  |   Glove    | Gensim |
| :--: | :-----------: | :-----------: | :-----------: | :----: | :--------: | :----: |
| Acc  |    0.7277     |    0.7393     |    0.7450     | 0.7441 | **0.7513** | 0.7469 |
|  F1  |    0.7225     |    0.7342     |    0.7400     | 0.7399 | **0.7470** | 0.7426 |

#### 实验结论

- 预训练的词向量Glove > gensim 自己训练词向量 > 随机初始化，收敛速度也在递减
- 词向量初始化为均匀分布效果似乎要比标准正态分布要好，起码收敛速度加快
- Rand1分别尝试了epoch为2、4、8，模型最终仍未完全收敛。其余epoch均为2，模型收敛程度稍好。