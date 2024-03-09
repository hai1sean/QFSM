# QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition with Minimal Gated Unit

A quantum federated learning algorithm for speech emotion recognition, designed using minimal gated units in quantum models.

Constructed by integrating the quantum model QMGU and quantum federated framework into the [classical SER methods](https://github.com/Renovamen/Speech-Emotion-Recognition.git).

The structural design of the [classical MGU](https://link.springer.com/article/10.1007/s11633-016-1006-2) can be seen in the literature:
>[MGU] Zhou G B, Wu J, Zhang C L, et al. Minimal gated unit for recurrent neural networks[J]. International Journal of Automation and Computing, 2016, 13(3): 226-234.

Compared to [QLSTM](https://github.com/rdisipio/qlstm.git) and [QGRU](https://github.com/zhenhouhong/QSpeech.git), QMGU shows a significant improvement in training speed due to the use of fewer VQC modules.

English Document | [中文文档](README.md)

&nbsp;

## Environments

- Python 3.7
- [PennyLane](https://pennylane.ai/)   a quantum simulation platform

&nbsp;

## Structure

```
├── configs/                // Configuration parameters (.yaml)
│   └── lstm.yaml           // Use this configuration file for path and parameter settings
├── models/                // Model implementations
│   ├── dnn                // Neural network models
│       └── lstm.py        // Use this file for quantum model construction
├── extract_feats/         // Feature extraction
│   └── opensmile.py       // Use Opensmile for feature extraction
├── utils/                 // Auxiliary tools
├── features/              // Store extracted features here, paths set in lstm.yaml
├── checkpoints/           // Store trained model weights
├── PrintLogs/             // Print evaluation metrics during training
├── example_pos.py         // Used for training
├── qlstm_pennylane.py     // Responsible for building QLSTM, QGRU, QMGU quantum circuits
└── preprocess.py          // Data preprocessing (extract features from audio data and save)
```
Other `.py` files named with "draw" can be used to plot curves, bar charts, heat maps for reference.

&nbsp;

## Tools

- [Opensmile](https://github.com/naxingyu/opensmile): Feature extraction, using the IS10_paraling settings yields 1582-dimensional features

&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   English, this dataset was developed by the audio research group at Ryerson University in Canada and provided by the Emotional Analysis Lab, with 24 professional speakers. The speech emotion section includes 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised. Each speaker recorded 60 samples, totaling 1440 different English speech samples.

2. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   German, a German speech emotion dataset created by the speech research group at the Berlin Institute of Technology in Germany, with 10 professional speakers covering 7 different emotions: neutral, angry, fear, happy, sad, disgust, boredom. It consists of 535 German speech samples.

3. CASIA

   Chinese speech emotion dataset recorded by the Institute of Automation, Chinese Academy of Sciences, including four professional speakers and six emotions: anger, happiness, fear, sadness, surprise, and neutral. It includes 9,600 different utterances, including 300 identical texts and 100 different texts. The 300 identical texts are available for free, totaling 1200 audio files for use.

&nbsp;

## Usage

### Preprocess

First, extract features from the audio data in the dataset and save them locally. Features extracted by Opensmile will be saved in `.csv` files.

For model accuracy, preprocess the dataset by recommending splitting each audio into 5 equally sized segments with 30% frame overlap between adjacent audio subsegments.

Due to the small data volume, data augmentation is suggested by adding Gaussian noise with a mean of 0 and standard deviation of 0.002 to improve the training effectiveness of the quantum model.

The extracted `.csv` files will be large, exceeding 100MB, so they cannot be uploaded here.

Since each set of 5 data comes from the same audio, handle the new dimension 5 carefully in `load_feature` of `opensmile.py`.

&nbsp;

### Train

Use `example_pos.py` for training.

Data is partitioned based on the number and ratio of clients.

Non-federated code is commented out and can be used.

Visualization of accuracy and loss using TensorBoard commands is also commented out.

&nbsp;

### Quantum model

All three quantum models are in `qlstm_pennylane.py`.

Simplest circuits are used for all VQCs, such as this:
```
qml.AngleEmbedding(inputs, wires=self.wires_forget, rotation='X')
qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
```

You can try the 19 circuit configurations mentioned in the literature [^1], but in the end, we found that it did not improve the accuracy; it only consumed more time. (It's possible that my dataset was still too small, but I consider this possibility to be quite low.)

QMGU is much faster because it uses one less VQC (run only on CPU using PennyLane) and the accuracy is not compromised.

The essence lies in these two lines, compared with QLSTM and QGRU, neither of which can omit a VQC:
```
f_t = torch.sigmoid(self.layer_out(self.forgetlayer(y_t)))  # forget block
n_t = torch.tanh(self.layer_out(self.block1layer(xq_t)) + f_t * self.layer_out(self.block2layer(hq_t)))  # new block
```

&nbsp;

## Citation
Quantum SER results are currently scarce, and the two QLSTM papers [^2][^3] did not use such large datasets. QGRU [^4] focuses on classical aspects.

We applied the QFSM algorithm to IoV, integrating it with the quantum federated learning framework. The parameters of the quantum model can be directly obtained from the code implementation, but attention should be paid to the way they are copied.

For more experimental results and details, please refer to our paper published in T-IV: [QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition with Minimal Gated Unit in 5G IoV](https://ieeexplore.ieee.org/abstract/document/10453624). We hope this document is helpful to you!
> [QFSM] Z. Qu, Z. Chen, S. Dehdashti and P. Tiwari, "QFSM: A Novel Quantum Federated Learning Algorithm for Speech Emotion Recognition With Minimal Gated Unit in 5G IoV," in IEEE Transactions on Intelligent Vehicles, doi: 10.1109/TIV.2024.3370398.

[^1]: Sim S, Johnson P D, Aspuru - Guzik A. Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms[J]. Advanced Quantum Technologies, 2019, 2(12): 1900070.
[^2]: Di Sipio R, Huang J H, Chen S Y C, et al. The dawn of quantum natural language processing[C]. ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 8612-8616.
[^3]: Chen S Y C, Yoo S, Fang Y L L. Quantum long short-term memory[C]. ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 8622-8626.
[^4]: Hong Z, Wang J, Qu X, et al. QSpeech: Low-Qubit Quantum Speech Application Toolkit[C]. 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022: 1-8.
