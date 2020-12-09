# standard module imports

# third party imports
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import splev, splrep

# team module imports
from QuadraticFit import QuadraticFit
from AitasInterface import AitasInterface


class AitasConcrete(AitasInterface):
    def __init__(self, input_stream: cv.VideoCapture, camera_gps: np.ndarray):
        super(AitasConcrete, self).__init__(input_stream, camera_gps)
        self.quad_fit = QuadraticFit()
        # Kernel Regression
        # todo: predefine hyperparameters
        self.lat_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)  # define rbf kernel
        self.lon_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)  # define rbf kernel

    def detect_bounding_box(self, frame):
        boxes, confs, clss, status_flag = self.trt_yolo.detect(frame, self.conf_th)
        if status_flag:
            x_tl = boxes[0, 0]
            y_tl = boxes[0, 1]
            x_br = boxes[0, 2]
            y_br = boxes[0, 3]

            x = (x_br + x_tl) / 2
            y = (y_br + y_tl) / 2

            # ENTER HORIZON CODE
            # a_norm = (x + (0.5 * w)) / self.a_lim
            b_norm = (self.b_lim - x)/self.b_lim
            a_norm = (y / self.a_lim)
            frame_box = np.array([a_norm, b_norm, confs])
            return frame_box, status_flag
        return boxes,status_flag

    def infer_gps_location(self, normalized_image_coordinates):
        a_norm = normalized_image_coordinates[0]
        b_norm = normalized_image_coordinates[1]
        confs = normalized_image_coordinates[2]

        cam_lat = self.camera_gps[0]
        cam_lon = self.camera_gps[1]
        lat_norm, lon_norm = self.quad_fit.quad_predict(a_norm, b_norm, cam_lat, cam_lon, self.SCALE_FACTOR)
        normalized_gps_coords = np.array([lat_norm, lon_norm])
        return normalized_gps_coords

    def interp_extrap_normalized_gps_data(self,n_interp,n_extrap=0):
        """Input: n_interp is the number of required trajectoty points
            Output: interpolated_data: n_data normalized GPS coordinates
        """
        # filtering data by confidence
        confid = (np.array(self.normalized_gps_df['Confs']))[:, np.newaxis]
        filtered_entries = (confid >= 0.75 ).all(axis=1)
        df_confid_filtered = self.normalized_gps_df[filtered_entries]

        # filtering data by z-score
        z_scores = stats.zscore(df_confid_filtered)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 2 ).all(axis=1)
        df_filtered = df_confid_filtered[filtered_entries]

        # extract data
        # ['Time', 'Norm_Lats', 'Norm_Longs', 'Confs']
        time_data = (df_filtered['Time'].values) * self.sampling_time
        if time_data[0] >= self.sampling_time:
            time_data = time_data - self.sampling_time
        lat_data = (df_filtered['Norm_Lats'].values)
        lon_data = (df_filtered['Norm_Longs'].values)
        weights = (df_filtered['Confs'].values)
        

        # Interpolation
        time_data_interp = (self.normalized_gps_df['Time'].values) * self.sampling_time
        if time_data_interp[0] >= self.sampling_time:
            time_data_interp = time_data_interp - self.sampling_time
        interp_time = np.linspace(time_data_interp[0], time_data_interp[-1], 
                                    num=n_interp, endpoint=True)
               

        lat_fcn = splrep(time_data, lat_data)
        lon_fcn = splrep(time_data, lon_data)

        lat_interp = splev(interp_time, lat_fcn)  # predict lat
        lon_interp = splev(interp_time, lon_fcn)  # predict lon

        interpolated_data = pd.DataFrame.from_dict({'normalized_lat': lat_interp, 'normalized_lon': lon_interp})
        extrapolated_data = []  # todo: implement extrapolation

        return interpolated_data, extrapolated_data
