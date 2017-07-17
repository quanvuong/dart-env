__author__ = 'yuwenhao'

from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
import numpy as np

class DynamicModel:
    def fit(self, X, Y):
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
        return np.array([np.concatenate([state_act, state_act**2, [1]])])

    def fit(self, X, Y):
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

    def fit(self, X, Y):
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

    def fit(self, X, Y):
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