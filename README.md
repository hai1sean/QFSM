# QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition with Minimal Gated Unit

一个使用量子最小门控单元设计的，用于语音情感识别的，量子联邦学习算法

由 [经典SER方法](https://github.com/Renovamen/Speech-Emotion-Recognition.git) 添加量子模型QMGU与量子联邦框架后构建

[经典MGU](https://link.springer.com/article/10.1007/s11633-016-1006-2)的结构设计可见文献 
>[MGU] Zhou G B, Wu J, Zhang C L, et al. Minimal gated unit for recurrent neural networks[J]. International Journal of Automation and Computing, 2016, 13(3): 226-234.

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
│   ├── files.py           
│   ├── opts.py            
│   └── plot.py            
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

   英文，该数据集由加拿大莱尔森大学（Ryerson University）的音频研究小组开发，并由情感研究实验室（Emotional Analysis Lab）提供，共有24个专业发音人，语音情感部分共有8种情绪，neutral, calm, happy, sad, angry, fearful, disgust, surprised，每个发音人录制60条，共1440条不同的英语发音样本。

2. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，德语语音情感数据集，该数据集由德国的柏林工业大学的语音学研究小组创建，共有10个专业发音人，涵盖了7种不同的情绪，包括neutral, angry, fear, happy, sad, disgust, boredom，共535个德语语音样本。

3. CASIA

   汉语语音情感数据集，中国科学院自动化研究所录制，共包括四个专业发音人，六种情绪，生气、高兴、害怕、悲伤、惊讶和中性，共9,600句不同发音，包括300句相同文本和100句不同文本。其中300句相同文本部分可以免费获取得到，共1200个音频可供使用。


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

所有VQC都用最简单的线路 比如这样 可以按照文献[^1]中的19种线路去尝试 最后发现并不能提升准确率 只会更费时间 （有可能是我的数据量还是太少了，但我认为这个可能性很小）
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

## Citation
量子SER目前成果很少，两篇QLSTM[^2][^3]都没有用规模那么大的数据集，QGRU[^4]重点在经典上

我们将这个QFSM算法用在IoV上，结合了量子联邦学习框架，量子模型的参数在代码实现上是可以直接拿出来的，但要注意copy的方式

更多实验结果与内容可以去查阅我们发表在T-IV上的论文 [QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition with Minimal Gated Unit in 5G IoV](https://ieeexplore.ieee.org/abstract/document/10453624)，希望这个文档对你有帮助！
> [QFSM] Z. Qu, Z. Chen, S. Dehdashti and P. Tiwari, "QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition With Minimal Gated Unit in 5G IoV," in IEEE Transactions on Intelligent Vehicles, doi: 10.1109/TIV.2024.3370398.

[^1]: Sim S, Johnson P D, Aspuru - Guzik A. Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms[J]. Advanced Quantum Technologies, 2019, 2(12): 1900070.
[^2]: Di Sipio R, Huang J H, Chen S Y C, et al. The dawn of quantum natural language processing[C]. ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 8612-8616.
[^3]: Chen S Y C, Yoo S, Fang Y L L. Quantum long short-term memory[C]. ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 8622-8626.
[^4]: Hong Z, Wang J, Qu X, et al. QSpeech: Low-Qubit Quantum Speech Application Toolkit[C]. 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022: 1-8.




