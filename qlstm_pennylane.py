import torch
import torch.nn as nn

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

class QLSTM(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                n_qubits=4,
                n_qlayers=1,
                n_vrotations=3,
                batch_first=True,
                return_sequences=False,
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        #self.dev = qml.device("default.qubit", wires=self.n_qubits)
        #self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        #self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def ansatz(params, wires_type):
            # Entangling layer.
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i - self.n_qubits]])

            # Variational layer.
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            # qml.templates.AngleEmbedding(features, wires=wires_type)
            ry_params = [torch.arctan(feature) for feature in features]
            rz_params = [torch.arctan(feature ** 2) for feature in features]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(rz_params[i], wires=wires_type[i])

            # Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            # VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            # VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            # VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            # VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        # weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        # self.clayer_in = torch.nn.Sequential(
        #     nn.Linear(self.concat_size, 100),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(100, self.n_qubits)
        # )
        # self.VQC = {
        #     'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
        #     'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
        #     'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
        #     'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        # }

        self.forgetlayer = qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes)
        self.inputlayer = qml.qnn.TorchLayer(self.qlayer_input, weight_shapes)
        self.updatelayer = qml.qnn.TorchLayer(self.qlayer_update, weight_shapes)
        self.outputlayer = qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)

        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = torch.nn.Sequential(
        #     nn.Linear(self.n_qubits, self.hidden_size),
        #     # nn.ReLU(True),
        #     nn.Dropout(p=0.5)
        # )
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)
            # y_t = v_t

            f_t = torch.sigmoid(self.clayer_out(self.forgetlayer(y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.inputlayer(y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.updatelayer(y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.outputlayer(y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class QGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 n_vrotations=3,
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit"):
        super(QGRU, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_block_1 = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_block_2 = [f"wire_forget_{i}" for i in range(self.n_qubits)]


        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_block_1 = qml.device(self.backend, wires=self.wires_block_1)
        self.dev_block_2 = qml.device(self.backend, wires=self.wires_block_2)


        self.layer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.nlayer_xin = torch.nn.Linear(self.n_inputs, n_qubits)
        self.nlayer_hin = torch.nn.Linear(self.hidden_size, n_qubits)
        self.layer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            # VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            # VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_block_1(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_block_1)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_block_1)
            # VQC(inputs, weights, self.wires_block_1)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_block_1]
        self.qlayer_block_1 = qml.QNode(_circuit_block_1, self.dev_block_1, interface="torch")

        def _circuit_block_2(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_block_2)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_block_2)
            # VQC(inputs, weights, self.wires_block_2)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_block_2]
        self.qlayer_block_2 = qml.QNode(_circuit_block_2, self.dev_block_2, interface="torch")

        weight_shapes = {"weights": (self.n_qlayers, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.forgetlayer = qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes)
        self.updatelayer = qml.qnn.TorchLayer(self.qlayer_update, weight_shapes)
        self.block1layer = qml.qnn.TorchLayer(self.qlayer_block_1, weight_shapes)
        self.block2layer = qml.qnn.TorchLayer(self.qlayer_block_2, weight_shapes)

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.layer_in(v_t)
            xq_t = self.nlayer_xin(x_t)
            hq_t = self.nlayer_hin(h_t)

            r_t = torch.sigmoid(self.layer_out(self.forgetlayer(y_t)))  # reset block
            z_t = torch.sigmoid(self.layer_out(self.updatelayer(y_t)))  # update block
            n_t = torch.tanh(self.layer_out(self.block1layer(xq_t)) + r_t * self.layer_out(self.block2layer(hq_t)))  # new block

            h_t = ((1 - z_t) * n_t) + (z_t * h_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

def vqc(n_qubits, n_qlayers, qembed_type="angle", qlayer_type="basic"):

    dev = qml.device("default.qubit", wires=n_qubits)
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    def _circuit(inputs, weights):
        # setting embedding
        if "angle" == qembed_type:
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        if "amplitude" == qembed_type:
            qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits))
        if "basic" == qembed_type:
            qml.templates.BasisEmbedding(inputs, wires=range(n_qubits))
        # setting layer
        if "basic" == qlayer_type:
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        if "strong" == qlayer_type:
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    qlayer = qml.QNode(_circuit, dev, interface="torch")

    weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}

    return qml.qnn.TorchLayer(qlayer, weight_shapes)

class QMGU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 n_vrotations=3,
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit"):
        super(QMGU, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_block_1 = [f"wire_block_1_{i}" for i in range(self.n_qubits)]
        self.wires_block_2 = [f"wire_block_2_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_block_1 = qml.device(self.backend, wires=self.wires_block_1)
        self.dev_block_2 = qml.device(self.backend, wires=self.wires_block_2)

        def ansatz(params, wires_type):
            # # Entangling layer.
            # for i in range(1, 3):
            #     for j in range(self.n_qubits):
            #         if j + i < self.n_qubits:
            #             qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
            #         else:
            #             qml.CNOT(wires=[wires_type[j], wires_type[j + i - self.n_qubits]])
            #
            # # Variational layer.
            # for i in range(self.n_qubits):
            #     qml.RX(params[0][i], wires=wires_type[i])
            #     qml.RY(params[1][i], wires=wires_type[i])
            #     qml.RZ(params[2][i], wires=wires_type[i])
            qml.CRX(params[0], [wires_type[self.n_qubits - 1], wires_type[0]])
            for i in range(self.n_qubits-1, 0, -1):
                qml.CRX(params[i], [wires_type[i-1], wires_type[i]])


        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            qml.templates.AngleEmbedding(features, wires=wires_type)
            # rx_params = [torch.arctan(feature) for feature in features]
            # rz_params = [torch.arctan(feature ** 2) for feature in features]
            # for i in range(self.n_qubits):
            #     # qml.Hadamard(wires=wires_type[i])
            #     qml.RX(rx_params[i].item(), wires=wires_type[i])
            #     qml.RZ(rz_params[i].item(), wires=wires_type[i])

            # Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        # p=0.1
        # K0=np.sqrt(1-p)*np.eye(2)
        # K1=np.sqrt(p)*np.array([[0,1j],[1j,0]])



        def _circuit_forget(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_forget, rotation='X')
            qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
            # for w in self.wires_forget:
            #     # qml.BitFlip(0.1, wires=w)
            #     qml.QubitChannel([K0,K1],wires=w)

            # VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        # print(qml.draw(self.qlayer_forget)(torch.zeros(4), torch.zeros((2, 4, 3))))

        def _circuit_block_1(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_block_1, rotation='X')
            qml.BasicEntanglerLayers(weights, wires=self.wires_block_1)
            # for w in self.wires_block_1:
            #     # qml.BitFlip(0.1, wires=w)
            #     qml.QubitChannel([K0, K1], wires=w)
            # VQC(inputs, weights, self.wires_block_1)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_block_1]
        self.qlayer_block_1 = qml.QNode(_circuit_block_1, self.dev_block_1, interface="torch")

        def _circuit_block_2(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_block_2, rotation='X')
            qml.BasicEntanglerLayers(weights, wires=self.wires_block_2)
            # for w in self.wires_block_2:
            #     # qml.BitFlip(0.1, wires=w)
            #     qml.QubitChannel([K0, K1], wires=w)
            # VQC(inputs, weights, self.wires_block_2)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_block_2]
        self.qlayer_block_2 = qml.QNode(_circuit_block_2, self.dev_block_2, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        # weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.layer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.nlayer_xin = torch.nn.Linear(self.n_inputs, n_qubits)
        self.nlayer_hin = torch.nn.Linear(self.hidden_size, n_qubits)

        # self.VQC = {
        #     'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
        #     'block_1': qml.qnn.TorchLayer(self.qlayer_block_1, weight_shapes),
        #     'block_2': qml.qnn.TorchLayer(self.qlayer_block_2, weight_shapes)
        # }
        self.forgetlayer = qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes)
        self.block1layer = qml.qnn.TorchLayer(self.qlayer_block_1, weight_shapes)
        self.block2layer = qml.qnn.TorchLayer(self.qlayer_block_2, weight_shapes)

        self.layer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)


    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.layer_in(v_t)
            xq_t = self.nlayer_xin(x_t)
            hq_t = self.nlayer_hin(h_t)


            f_t = torch.sigmoid(self.layer_out(self.forgetlayer(y_t)))  # forget block
            n_t = torch.tanh(self.layer_out(self.block1layer(xq_t)) + f_t * self.layer_out(self.block2layer(hq_t)))  # new block
            # z_t = torch.sigmoid(self.layer_out(self.VQC[1](y_t)))  # update block
            # n_t = torch.tanh(self.layer_out(self.VQC[2](xq_t)) + r_t * self.layer_out(self.VQC[3](hq_t)))  # new block

            h_t = ((1 - f_t) * h_t) + (f_t * n_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
