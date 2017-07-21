__author__ = 'yuwenhao'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import numpy as np
from rllab.core.network import MLP
import lasagne.nonlinearities as NL
import theano.tensor as TT
import theano as T
import lasagne
import lasagne.layers as L
import theano
from rllab.misc import ext
import numpy as np

class DynamicModel:
    def fit(self, X, Y, iter=1):
        raise NotImplementedError

    def do_simulation(self, state_vec, tau, frame_skip):
        raise NotImplementedError

    def dfdx(self, state, act):
        raise NotImplementedError

    def dfda(self, state, act):
        raise NotImplementedError

    def do_simulation_corrective(self, state_vec, tau, frame_skip, next_state, ds, da):
        return next_state + ds.dot(self.dfdx(state_vec, tau)) + da.dot(self.dfda(state_vec, tau))


class LinearDynamicModel(DynamicModel):
    def __init__(self):
        self._coeffs = None
        self._reg_coeff = 1e-5

    def _features(self, state_act):
        return np.array([np.concatenate([state_act, state_act**2, np.sin(state_act), np.cos(state_act), [1]])])

    def fit(self, X, Y, iter=1):
        self.state_dim = len(Y[0])
        featmat = np.concatenate([self._features(state_act) for state_act in X])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(Y)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def do_simulation(self, state_vec, tau, frame_skip):
        if self._coeffs is None:
            return np.zeros(len(state_vec))

        return state_vec + self._features(np.concatenate([state_vec, tau])).dot(self._coeffs)[0]

    def dfdx(self, state, act):
        state_dim = len(state)
        act_dim = len(act)
        return np.identity(len(state)) + self._coeffs[0:state_dim, :] + self._coeffs[state_dim+act_dim:state_dim*2+act_dim, :] * np.vstack([state]*len(state)).T

    def dfda(self, state, act):
        state_dim = len(state)
        act_dim = len(act)
        return self._coeffs[state_dim:state_dim+act_dim, :] + self._coeffs[state_dim*2+act_dim:state_dim*2+act_dim*2, :] * np.vstack([act]*len(state)).T


class GPDynamicModel(DynamicModel):
    def __init__(self):
        k1 = 5.0**2 * RBF(length_scale=5.0)  # long term smooth rising trend
        kernel = k1
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer=None, normalize_y=True)

    def _features(self, state_act):
        return np.array([np.concatenate([state_act, state_act**2, [1]])])

    def fit(self, X, Y, iter=1):
        featmat = np.concatenate([self._features(state_act) for state_act in X])
        '''reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(Y)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10'''
        self.gp.fit(featmat, Y)

    def do_simulation(self, state_vec, tau, frame_skip):
        '''if self._coeffs is None:
            return np.zeros(len(state_vec))

        return self._features(np.concatenate([state_vec, tau])).dot(self._coeffs)[0]'''
        return self.gp.predict(self._features(np.concatenate([state_vec, tau])), return_std=False)[0]

class KNNDynamicModel(DynamicModel):
    def __init__(self):
        self._coeffs = None
        self._reg_coeff = 1e-5
        self.knn = KNeighborsRegressor(n_neighbors=5, weights='distance')

    def _features(self, state_act):
        return np.array([np.concatenate([state_act, state_act**2, [1]])])

    def fit(self, X, Y, iter=1):
        featmat = np.concatenate([self._features(state_act) for state_act in X])
        '''reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(Y)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10'''
        self.knn.fit(featmat, Y)

    def do_simulation(self, state_vec, tau, frame_skip):
        '''if self._coeffs is None:
            return np.zeros(len(state_vec))

        return self._features(np.concatenate([state_vec, tau])).dot(self._coeffs)[0]'''
        return self.knn.predict(self._features(np.concatenate([state_vec, tau])))[0]

class MLPDynamicModel(DynamicModel):
    def __init__(self):
        self.mlp = None
        self.train_fn = None
        self.loss = None
        self.out = None
        self.grad_dsa_fn = None

    def _features(self, state_act):
        return np.array([np.concatenate([state_act, state_act**2, [1]])])

    def fit(self, X, Y, iter=1):
        if self.mlp is None:
            self.mlp = MLP(
                input_shape=(len(X[0]),),
                output_dim=len(Y[0]),
                hidden_sizes=(32,32),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=None,)
            out_var = TT.matrix('out_var')
            prediction = self.mlp._output
            loss = lasagne.objectives.squared_error(prediction, out_var)
            loss = loss.mean()
            params = self.mlp.get_params(trainable=True)
            updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
            self.train_fn = T.function([self.mlp.input_layer.input_var, out_var], loss, updates=updates)
            self.loss = T.function([self.mlp.input_layer.input_var, out_var], loss)
            self.out = T.function([self.mlp.input_layer.input_var], prediction)
            grad_dsa = theano.gradient.jacobian(TT.flatten(prediction), wrt=self.mlp.input_layer.input_var, disconnected_inputs='warn')
            self.grad_dsa_fn = ext.compile_function(
                inputs=[self.mlp.input_layer.input_var],
                outputs=grad_dsa,
                log_name="f_grad",
            )

        for epoch in range(iter):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            for batch in self.iterate_minibatches(X, Y, 32, shuffle=True):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1
            # Then we print the results for this epoch:
            #if epoch%10 == 9:
            print("aux training loss:\t\t{:.6f}".format(train_err / train_batches))


    def do_simulation(self, state_vec, tau, frame_skip):
        return state_vec + self.out([np.concatenate([state_vec, tau])])[0]

    def do_simulation_corrective(self, state_vec, tau, frame_skip, next_state, ds, da):
        dfdsa = self.grad_dsa_fn([np.concatenate([state_vec, tau])])[0]
        return next_state + dfdsa.dot(np.concatenate([ds, da])).T

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield np.array(inputs)[excerpt], np.array(targets)[excerpt]

