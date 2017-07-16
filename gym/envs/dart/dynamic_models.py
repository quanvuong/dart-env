__author__ = 'yuwenhao'

from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class DynamicModel:
    def fit(self, X, Y):
        raise NotImplementedError

    def do_simulation(self, state_vec, tau, frame_skip):
        raise NotImplementedError

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

        return self._features(np.concatenate([state_vec, tau])).dot(self._coeffs)[0]

class KNNDynamicModel(DynamicModel):
    def __init__(self):
        self._coeffs = None
        self._reg_coeff = 1e-5
        self.knn = KNeighborsRegressor(n_neighbors=1, weights='distance')

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