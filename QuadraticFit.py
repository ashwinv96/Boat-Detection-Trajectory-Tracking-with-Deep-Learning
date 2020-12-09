from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq


class QuadraticFit(object):
    def __init__(self, *args):
        if len(args) == 2:
            self.q_lat = args[0]
            self.q_lon = args[1]
        elif len(args) == 1:
            try:
                # open directory
                directory = args[0]
                self.q_lat = np.load(f'{directory}/q_lat.npy')
                self.q_lon = np.load(f'{directory}/q_lon.npy')
            except Exception as e:
                raise e

        else:
            try:
                # open directory
                self.q_lat = np.load(f'weights/q_lat.npy')
                self.q_lon = np.load(f'weights/q_lon.npy')
            except FileNotFoundError:
                self.q_lat = np.empty((6, 1), dtype=np.float64)
                self.q_lon = np.empty((6, 1), dtype=np.float64)

    @classmethod
    def vec(cls, v_matrix):
        n = v_matrix.shape[0]  # implement a catch here to validate Symmetricity of matrix
        v = np.zeros((int(0.5 * n * (n + 1)), 1))  # initializing v
        start_index = 0

        for index in range(n):
            end_index = start_index + (n - index)
            v[start_index:end_index] = np.diag(v_matrix, index)[:, np.newaxis]
            start_index = end_index

        return v

    #
    # def set_weight(self, q_lat, q_lon):
    #     self.q_lat = q_lat
    #     self.q_lon = q_lon

    def quad_predict(self, a_norm, b_norm, cam_lat, cam_lon, scale_factor):
        # packing input
        x = np.array([a_norm, b_norm])

        # calling quad_fun
        delta_lat_quad = self.quad_fun(x, self.q_lat)
        delta_lon_quad = self.quad_fun(x, self.q_lon)
        # print(delta_lat_quad, delta_lon_quad)

        # # un-normalizing data
        #lat = cam_lat + delta_lat_quad / scale_factor
        #lon = cam_lon + delta_lon_quad / scale_factor
        return delta_lat_quad, delta_lon_quad

    @classmethod
    def quad_fun(cls, x, q):
        # convert x to column vector
        if type(x) == np.ndarray:
            if len(x.shape) == 1:
                x = x[:, np.newaxis]  # for 1-D array
            elif x.shape[0] == 1:
                x = x.T  # for n-D array

        tmp = np.vstack((x, 1))
        Z = np.dot(tmp, tmp.T)
        z = cls.vec(Z)

        y = np.dot(q.T, z)
        return y

    @classmethod
    def train_quad(cls, X_matrix, y_vec):
        # unpack sizes
        n, N, *tmp = X_matrix.shape

        # initialize
        m = int(0.5 * (n + 1) * (n + 2))
        A = np.zeros((m, m))
        b = np.zeros((m, 1))

        for index in range(N):
            # current input-output sample
            if len(y_vec.shape) == 1:
                y_i = y_vec[index]
                x_i = X_matrix[:, index][:, np.newaxis]
            else:
                try:
                    # row vector
                    y_i = y_vec[0, index]
                    x_i = X_matrix[:, index][:, np.newaxis]
                except IndexError:
                    # column vector (x_i)
                    y_i = y_vec[index, 0]
                    x_i = X_matrix[:, index][:, np.newaxis]

            # feature calculation
            tmp = np.vstack((x_i, 1))
            Z_i = np.dot(tmp, tmp.T)  # Z_i = [x_i;1]*[x_i;1].'
            z_i = cls.vec(Z_i)

            # operator updates
            A = A + np.dot(z_i, z_i.T)  # A = A + z_i * z_i.'
            b = b + (y_i * z_i)

        # Solve for weights in the feature space
        q, res, rank, s = lstsq(A, b)  # or solve(A, b)

        return q, res, rank, s

    def train_weights(self, X, y_lat, y_lon):
        q_lat, _, rank1, _ = self.train_quad(X, y_lat)
        q_lon, _, rank2, _ = self.train_quad(X, y_lon)

        self.q_lat = q_lat
        self.q_lon = q_lon

    def save_weights(self, dir_name):
        # todo: handle exception caused by dir_name not existing
        dir_object = Path(dir_name)
        if not dir_object.exists():
            dir_object.mkdir()
        np.save(f'{dir_name}/q_lat', self.q_lat)
        np.save(f'{dir_name}/q_lon', self.q_lon)

    @staticmethod
    def remove_outliers(DataFrame, low_thresh, high_thresh):
        # y = df['rando']
        # removed_outliers = y.between(y.quantile(.05), y.quantile(.95))
        col = 0
        num_cols = len(DataFrame)
        mask = pd.Series([True] * num_cols, dtype=bool)
        for col in DataFrame:
            if col != 'conf':
                mask = mask.values & DataFrame[col].between(DataFrame[col].quantile(low_thresh), DataFrame[col].quantile(high_thresh))
                # filtered_entries = (z_scores >= 0.75).all(axis=1)
                # old_df = df[filtered_entries]
        return mask
