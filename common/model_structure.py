from collections import namedtuple
from typing import Any
import json, zipfile
import numpy as np
from tensorflow.keras import models, layers

RZModelData = namedtuple("RZModelData", ("models", "n_theta"))


def load_k2rz_data(model_path: str, n_models: int = 1, n_theta: int = 64):
    models_ = tuple(models.load_model(model_path + f'/best_model{i}', compile=False) for i in range(n_models))
    return RZModelData(models_, n_theta)


def predict_k2rz(data: RZModelData, model_indices: list[int], x, n_theta=None, closed_surface=True, xpt_correction=True):
    n_theta = n_theta if n_theta is not None else data.n_theta
    y = np.mean([data.models[i].predict(np.atleast_2d(x))[0] for i in model_indices], axis=0)
    rbdry, zbdry = y[:n_theta], y[n_theta:]

    if xpt_correction:
        rgeo, amin = 0.5 * (max(rbdry) + min(rbdry)), 0.5 * (max(rbdry) - min(rbdry))
        _, _, _, _, _, k, du, dl = x
        if du <= dl:
            rx = rgeo - amin * dl
            zx = max(zbdry) - 2 * k * amin
            rx2 = rgeo - amin * du
            rbdry[np.argmin(zbdry)] = rx
            zbdry[np.argmin(zbdry)] = zx
            rbdry[np.argmax(zbdry)] = rx2
        else:
            rx = rgeo - amin * du
            zx = min(zbdry) + 2 * k * amin
            rx2 = rgeo - amin * dl
            rbdry[np.argmax(zbdry)] = rx
            zbdry[np.argmax(zbdry)] = zx
            rbdry[np.argmin(zbdry)] = rx2
    
    if closed_surface:
        rbdry, zbdry = np.append(rbdry, rbdry[0]), np.append(zbdry, zbdry[0])

    return rbdry, zbdry


class x2rz():
    def __init__(self, model_path, n_models=1, ntheta=64, closed_surface=True, xpt_correction=True):
        self.nmodels, self.ntheta = n_models, ntheta
        self.closed_surface, self.xpt_correction = closed_surface, xpt_correction
        self.models = [models.load_model(model_path + f'/best_model{i}', compile=False) for i in range(self.nmodels)]

    def set_inputs(self, ip, bt, βp, rx1, zx1, rx2, zx2, drsep, rin, rout):
        self.x = np.array([ip, bt, βp, rx1, zx1, rx2, zx2, drsep, rin, rout])

    def predict(self, post=True):
        self.y = np.mean([m.predict(np.array([self.x]))[0] for m in self.models[:self.nmodels]], axis=0)
        rbdry, zbdry = self.y[:self.ntheta], self.y[self.ntheta:]
        if post:
            if self.xpt_correction:
                rgeo, amin = 0.5 * (max(rbdry) + min(rbdry)), 0.5 * (max(rbdry) - min(rbdry))
                if self.x[7] <= 0: # LSN
                    rbdry[np.argmin(zbdry)] = self.x[3]
                    zbdry[np.argmin(zbdry)] = self.x[4]
                else: # USN
                    rbdry[np.argmax(zbdry)] = self.x[5]
                    zbdry[np.argmax(zbdry)] = self.x[6]

            if self.closed_surface:
                rbdry, zbdry = np.append(rbdry, rbdry[0]), np.append(zbdry, zbdry[0])

        return rbdry, zbdry

def load_custom_model(input_shape, lstms, denses, model_path):
    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape = input_shape))
    for i, n in enumerate(lstms):
        rs = False if i == len(lstms) - 1 else True
        model.add(layers.LSTM(n, return_sequences = rs))
        model.add(layers.BatchNormalization())
    for n in denses[:-1]:
        model.add(layers.Dense(n, activation = 'sigmoid'))
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(denses[-1], activation = 'linear'))
    model.load_weights(model_path)
    return model

class kstar_lstm():
    def __init__(self, model_path, n_models=1, ymean=None, ystd=None):
        self.nmodels = n_models
        if ymean is None:
            self.ymean = [1.30934765, 5.20082444, 1.47538417, 1.14439883]
            self.ystd  = [0.74135689, 1.44731883, 0.56747578, 0.23018484]
        else:
            self.ymean, self.ystd = ymean, ystd
        self.models = [load_custom_model((10, 21), [200, 200], [200, 4], model_path + f'/best_model{i}.h5') for i in range(self.nmodels)]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 3 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean([m.predict(self.x)[0] * self.ystd + self.ymean for m in self.models[:self.nmodels]], axis=0)
        return self.y


NormalizedModelData = namedtuple("NormalizedModelData", ("models", "y_mean", "y_std"))


def load_kstar_v220505_data(model_path: str, n_models: int = 1, length: int = 10, y_mean: Any = None, y_std: Any = None):
    if y_mean is None or y_std is None:
        y_mean = np.array([1.4361666, 5.275876, 1.534538, 1.1268075])
        y_std = np.array([0.7294007, 1.5010427, 0.6472052, 0.2331879])
    models_ = tuple(load_custom_model((length, 18), [100, 100], [50, 4], model_path + f'/best_model{i}.h5') for i in range(n_models))
    return NormalizedModelData(models_, y_mean, y_std)



def predict_kstar_v220505(data: NormalizedModelData, model_indices: list[int], x: Any):
    x = np.expand_dims(x, axis=tuple(range(3 - np.ndim(x))))
    y = np.mean([data.models[i].predict(x)[0] * data.y_std + data.y_mean for i in model_indices], axis=0)
    return y


def load_kstar_nn_data(model_path: str, n_models: int = 1, y_mean: Any = None, y_std: Any = None):
    if y_mean is None or y_std is None:
        y_mean = [1.22379703, 5.2361062,  1.64438005, 1.12040048]
        y_std  = [0.72255576, 1.5622809,  0.96563557, 0.23868018]
    models_ = tuple(models.load_model(model_path + f'/best_model{i}', compile=False) for i in range(n_models))
    return NormalizedModelData(models_, y_mean, y_std)


def predict_kstar_nn(data: NormalizedModelData, x: Any):
    x = np.atleast_2d(x)
    y = np.mean([m.predict(x)[0] * data.y_std + data.y_mean for m in data.models], axis=0)
    return y


class bpw_nn():
    def __init__(self, model_path, n_models=1):
        self.nmodels = n_models
        self.ymean = np.array([1.02158800e+00, 1.87408512e+05])
        self.ystd  = np.array([6.43390272e-01, 1.22543529e+05])
        self.models = [models.load_model(model_path + f'/best_model{i}', compile=False) for i in range(self.nmodels)]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean([m.predict(self.x)[0] * self.ystd + self.ymean for m in self.models[:self.nmodels]], axis=0)
        return self.y


def load_tf_dense_data(model_path: str, n_models: int = 1, y_mean: Any = 0.0, y_std: Any = 1.0):
    models_ =  tuple(models.load_model(model_path + f'/best_model{i}', compile=False) for i in range(n_models))
    return NormalizedModelData(models_, y_mean, y_std)


def predict_tf_dense(data: NormalizedModelData, model_indices: list[int], x: Any):
    x = np.atleast_2d(x)
    y = np.mean([data.models[i].predict(x)[0] * data.y_std + data.y_mean for i in model_indices], axis=0)
    return y


def actv(x, method):
    if method == 'relu':
        return np.max([np.zeros_like(x), x], axis=0)
    elif method == 'tanh':
        return np.tanh(x)
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif method == 'linear':
        return x

class SB2_model():
    def __init__(self, model_path, low_state, high_state, low_action, high_action, activation='relu', last_actv='tanh', norm=True, bavg=0.):
        zf = zipfile.ZipFile(model_path)
        data = json.loads(zf.read('data').decode("utf-8"))
        self.parameter_list = json.loads(zf.read('parameter_list').decode("utf-8"))
        self.parameters = np.load(zf.open('parameters'))
        self.layers = data['policy_kwargs']['layers'] if 'layers' in data['policy_kwargs'].keys() else [64, 64]
        self.low_state, self.high_state = low_state, high_state
        self.low_action, self.high_action = low_action, high_action
        self.activation, self.last_actv = activation, last_actv
        self.norm = norm
        self.bavg = bavg

    def predict(self, x, yold=None):
        xnorm = 2 * (x - self.low_state) / np.subtract(self.high_state, self.low_state) - 1 if self.norm else x
        ynorm = xnorm
        for i, layer in enumerate(self.layers):
            w, b = self.parameters[f'model/pi/fc{i}/kernel:0'], self.parameters[f'model/pi/fc{i}/bias:0']
            ynorm = actv(np.matmul(ynorm, w) + b, self.activation)
        w, b = self.parameters[f'model/pi/dense/kernel:0'], self.parameters[f'model/pi/dense/bias:0']
        ynorm = actv(np.matmul(ynorm, w) + b, self.last_actv)

        y = 0.5 * np.subtract(self.high_action, self.low_action) * (ynorm + 1) + self.low_action if self.norm else ynorm
        if yold is None:
            yold = x[:len(y)]
        y =  self.bavg * yold + (1 - self.bavg) * y
        return y

class SB2_ensemble():
    def __init__(self, model_list, low_state, high_state, low_action, high_action, activation='relu', last_actv='tanh', norm=True, bavg=0.):
        self.models = [SB2_model(model_path, low_state, high_state, low_action, high_action, activation, last_actv, norm, bavg) for model_path in model_list]

    def predict(self, x):
        ys = [m.predict(x) for m in self.models]
        return np.mean(ys, axis=0)


