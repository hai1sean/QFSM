# QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition with Minimal Gated Unit

一个使用量子最小门控单元设计的，用于语音情感识别的，量子联邦学习算法

由 [经典SER方法](https://github.com/Renovamen/Speech-Emotion-Recognition.git) 添加量子模型QMGU与量子联邦框架后构建

[经典MGU](https://link.springer.com/article/10.1007/s11633-016-1006-2)的结构设计可见文献 
>[1] Zhou G B, Wu J, Zhang C L, et al. Minimal gated unit for recurrent neural networks[J]. International Journal of Automation and Computing, 2016, 13(3): 226-234.

QMGU相比 [QLSTM](https://github.com/rdisipio/qlstm.git) 和 [QGRU](https://github.com/zhenhouhong/QSpeech.git) 由于使用了更少的VQC模块 训练速度提升十分明显

[English Document](README_EN.md) | 中文文档


&nbsp;

## Environments

- Python 3.8
- [PennyLane](https://pennylane.ai/)   一个量子模拟平台 


&nbsp;

## Structure

```
├── configs/                // 配置参数（.yaml）
│   └── lstm.yaml           // 借用这个配置文件进行路径和参数设置
├── models/                // 模型实现
│   ├── common.py          
│   ├── dnn                // 神经网络模型           
│   │   └── lstm.py        // 借用该文件进行量子模型构建
│   └── ml.py              
├── extract_feats/         // 特征提取        
│   └── opensmile.py       // 使用 Opensmile 提取特征
├── utils/                 // 辅助工具
│   ├── files.py           // 用于整理数据集（分类、批量重命名）
│   ├── opts.py            // 使用 argparse 从命令行读入参数
│   └── plot.py            // 绘图 
├── features/              // 存储提取好的特征可存储在此，路径通过lstm.yaml设置
├── checkpoints/           // 存储训练好的模型权重
├── PrintLogs/             // 打印训练时的评估指标
├── example_pos.py         // 用于训练
├── qlstm_pennylane.py     // 负责QLSTM,QGRU,QMGU的量子线路搭建
└── preprocess.py          // 数据预处理（提取数据集中音频的特征并保存）
```


&nbsp;

## Tools

- [Opensmile](https://github.com/naxingyu/opensmile)：提取特征 使用设置文件IS10_paraling 提取到的特征维度为1582


&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   英文，24 个人（12 名男性，12 名女性）的大约 1500 个音频，表达了 8 种不同的情绪（第三位数字表示情绪类别）：01 = neutral，02 = calm，03 = happy，04 = sad，05 = angry，06 = fearful，07 = disgust，08 = surprised。

2. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，10 个人（5 名男性，5 名女性）的大约 500 个音频，表达了 7 种不同的情绪（倒数第二个字母表示情绪类别）：N = neutral，W = angry，A = fear，F = happy，T = sad，E = disgust，L = boredom。

3. CASIA

   汉语，只能下载到部分。4 个人（2 名男性，2 名女性）的大约 1200 个音频，表达了 6 种不同的情绪：neutral，happy，sad，angry，fearful，surprised。


&nbsp;

## Usage

### Preprocess

首先需要提取数据集中音频的特征并保存到本地。Opensmile 提取的特征会被保存在 `.csv` 文件中。

为了模型准确性，需要对数据集预处理，建议对于单条音频进行5等长的切割，并且在相邻的音频子段之间添加30%的帧重叠

由于数据量过少，建议做数据增强，添加均值为0，标准差为0.002的高斯分布噪声，提升量子模型的训练效果

这样提取到的`.csv` 文件会很大，超过100M，这里没法上传了

切割之后每5条数据其实是源自同一个音频，所以在`opensmile.py`的`load_feature`中要谨慎处理新增的维度5


&nbsp;

### Train

`example_pos.py` 进行训练

开始前对数据进行了划分，按照客户端数量和比例

非联邦的代码打在了注释中可以使用

最后用tensorboard可视化acc和loss 指令也打在了注释中

&nbsp;

### Quantum model

三个量子模型都在`qlstm_pennylane.py`

所有VQC都用最简单的线路 比如这样
```
qml.AngleEmbedding(inputs, wires=self.wires_forget, rotation='X')
qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
```

QMGU因为少用了一个VQC 所以速度快很多 (pennylane只用cpu跑) 且准确率并没有掉 

精髓在这两行  可以和QLSTM QGRU对比 前两者都没有办法省掉一个VQC

```
f_t = torch.sigmoid(self.layer_out(self.forgetlayer(y_t)))  # forget block
n_t = torch.tanh(self.layer_out(self.block1layer(xq_t)) + f_t * self.layer_out(self.block2layer(hq_t)))  # new block
```

&nbsp;






