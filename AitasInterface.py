# standard module imports
from abc import ABC, abstractmethod

# third party imports
import cv2 as cv
import numpy as np
import pandas as pd

# team module imports
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
import matplotlib.pyplot as plt

class AitasInterface(ABC):
    """The Base Class that is inherited"""

    def __init__(self, input_stream: cv.VideoCapture, camera_gps: np.ndarray):
        # todo: add verbosity/display mode flag
        """Constructor for ClassName"""
        self.input_stream = input_stream
        self.camera_gps = camera_gps
        self.normalized_gps_df = pd.DataFrame(dtype=np.float64)
        self.sampling_time = 0.05  # todo: change to use FPS

        # initialize constants
        self.SCALE_FACTOR = 1000.0

        # yolo model specific constants
        self.target_name_dict = {0: 'TargetShip'}
        self.vis = BBoxVisualization(self.target_name_dict)
        self.conf_th = 0.5
        self.display_mode = False  # verbosity flag
        self.model = 'yolov4-416'
        self.model_height = 416
        self.model_width = 416
        self.model_shape = (self.model_height, self.model_width)
        self.category_number = 80
        self.trt_yolo = TrtYOLO(self.model, self.model_shape, self.category_number)

        self.a_lim = 720.0
        self.b_lim = 1280.0 / 2.0
        # todo: load the models here and save them as constants
        super(AitasInterface, self).__init__()

    # @abstractmethod
    def detect_bounding_box(self, frame) -> np.ndarray:
        # original parameters: (self, frame)
        # todo: enforce output to be a row vector of shape (3,)
        # return flag, bounding box
        pass

    # @abstractmethod
    def infer_gps_location(self, normalized_image_coordinates: np.ndarray) -> np.ndarray:
        # todo: enforce shape
        """
        Inputs: a, b, w
        Outputs: normalized_delta_lat, normalized_delta_lon
        Description: calculate the camera coordinates
        """
        pass

    # @abstractmethod
    def interp_extrap_normalized_gps_data(self, n_interp, n_extrap=0):
        """
        Inputs: n_interp, n_extrap are the equidistant number of points to interpolate and extrapolate respectively.
                A dataframe is the input.
        TODO: enforce the output structure below
        Outputs:
            interpolated_data: n_interp by 2 normalized predicted (interpolated) GPS coordinate
            extrapolated_data: n_extrap by 2 normalized predicted (forecasted) GPS coordinate
        """
        pass

    @staticmethod
    def output(interpolated):
        """
        Todo: discuss
        Inputs: interpolated, extrapolated
        Outputs: csv, video, plots
        """
        # interpolated_df = pd.DataFrame(interpolated)
        #extrapolated_df = pd.DataFrame(extrapolated)


        # save to the current directory
        interpolated_save_path = 'interpolated_data.csv'
        #extrapolated_save_path = 'extrapolated_data.csv'

        interpolated.to_csv(interpolated_save_path, index=False)
        #extrapolated_df.to_csv(extrapolated_save_path)

    def normalize_gps(self, gps: np.ndarray) -> np.ndarray:
        return self.SCALE_FACTOR * (gps - self.camera_gps)

    def unnormalize_gps(self, delta_gps_norm):
        delta_gps_unnorm = delta_gps_norm
        delta_gps_unnorm.iloc[:, 0] = delta_gps_unnorm.iloc[:, 0].values / self.SCALE_FACTOR + self.camera_gps[0]
        delta_gps_unnorm.iloc[:, 1] = delta_gps_unnorm.iloc[:, 1].values / self.SCALE_FACTOR + self.camera_gps[1]
        return delta_gps_unnorm

    def process_video(self, n_interp, n_extrap=0):
        """
        Output: DataFrame.index([time, x_value_norm, y_value_norm])
        """
        num_frames = self.input_stream.get(cv.CAP_PROP_FRAME_COUNT)  # get the total number of frames
        # todo: define a constant for the time gap. The FPS is valid
        normalized_gps_array = np.zeros((1, 4), dtype=np.float64)
        while self.input_stream.isOpened():
            ret, frame = self.input_stream.read()

            if not ret:
                break

            ''' 
            Calling bounding box detection
            ---Expected output: a, b, w
            ## a: the vertical distance from the horizon to the middle bottom of the bounding box
            ## b: the horizontal distance from the midline to the middle of the bounding box
            ## w: the corresponding detection weights (detection confidence)
            '''

            frame_box, detected = self.detect_bounding_box(frame)
            # detected, frame_box = self.detect_bounding_box(frame)  # outputs a row vector
            # todo: implement method to convert detected_bounding_box to [a, b]

            '''
            Calling GPS inference method (corresponding machine learning function that gets normalized gps coordinates
            from the current bounding box)
            '''

            if detected and len(frame_box[2]) == 1:
                frame_normalized_gps = self.infer_gps_location(frame_box)
                current_frame_number = self.input_stream.get(cv.CAP_PROP_POS_FRAMES)
                current_frame_number = np.array([current_frame_number], dtype=np.float64)
                #if current_frame_number > 50:
                    #break
                confs = frame_box[2]
                # todo: test that line below functions properly
                # todo: pass time as a column or index to the dataframe
                normalized_lats = frame_normalized_gps[0, 0]
                # normalized_lats = normalized_lats[0]
                normalized_longs = frame_normalized_gps[1, 0]
                # normalized_longs = normalized_longs[0]

                data_to_append = np.array([current_frame_number, normalized_lats, normalized_longs, confs]).T
                print(data_to_append)
                normalized_gps_array = np.append(normalized_gps_array, data_to_append, axis=0)
                
        self.input_stream.release()
        
        #outputting result in line
        interpolated_indices = (np.ceil(np.linspace(0, normalized_gps_array.shape[0] - 1, n_interp))).astype(int)
        #print(interpolated_indices)
        time_interp = normalized_gps_array[interpolated_indices, 0] * self.sampling_time
        time_interp[1] = 0.0
        lat_interp = normalized_gps_array[interpolated_indices, 1] / self.SCALE_FACTOR + self.camera_gps[0]
        lon_interp = normalized_gps_array[interpolated_indices, 2] / self.SCALE_FACTOR + self.camera_gps[1]
		
        column_names = ['Time', 'Latitude', 'Longitude']
        interpolated_gps = pd.DataFrame(np.concatenate((time_interp.reshape(-1,1), lat_interp.reshape(-1,1), lon_interp.reshape(-1,1)), axis=1), columns=column_names, dtype=np.float64) 
        #self.normalized_gps_df = pd.DataFrame(normalized_gps_array[1:], columns=column_names, dtype=np.float64)
        '''
        Call interp, extrap method and output the interpolated and extrapolated 
        '''
        #interpolated, extrapolated = self.interp_extrap_normalized_gps_data(n_interp, n_extrap)
        
        #extrapolated_df = pd.DataFrame(extrapolated)

        #t1 = (self.normalized_gps_df['Time'] - 1) * self.sampling_time

        #plt.plot(t1, self.normalized_gps_df.iloc[:, 0], 'o')
        #plt.plot(interpolated.iloc[:, 2], interpolated.iloc[:, 0])
        #plt.show()

        #plt.plot(t1, self.normalized_gps_df.iloc[:, 1], 'o')
        #plt.plot(interpolated.iloc[:, 2], interpolated.iloc[:, 1])
        #plt.show()

        #concat_df = pd.concat([self.normalized_gps_df, interpolated], ignore_index = True, axis = 1)
        #oncat_df.to_csv('concat_df.csv')

        # Un-normalize the GPS data
        
        #print(interpolated_indices)  
        #interpolated_gps = self.unnormalize_gps(self.normalized_gps_df.iloc[interpolated_indices])
        # todo: check if extrapolated_gps is empty
        #extrapolated_gps = self.unnormalize_gps(extrapolated_df)
        '''
        Output presentation/format: csv, plot, video
        '''
        # cv2.destroyAllWindows()  not necessary if nothing is being displayed
        
        AitasInterface.output(interpolated_gps.iloc[1:,:])
        
