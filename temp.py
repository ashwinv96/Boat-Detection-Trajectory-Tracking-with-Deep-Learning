# import cv2 as cv
#
# path_to_vid = 'data/6.mp4'
#
# stream = cv.VideoCapture(path_to_vid)
# numFrames = stream.get(cv.CAP_PROP_FRAME_COUNT)
# print(numFrames)
#
# while stream.isOpened():
#     ret, img = stream.read()
#     # display the current frame
#     # cv.imshow('Current Frame', img)
#     print(stream.get(1))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from QuadraticFit import QuadraticFit as qf


# from train_quad import train_quad
# from quad_fun import quad_fun
# from quad_predict import quad_predict


def AI_track_sea_cam2loc():
    # Looading data
    file_name = 'quad_train_dataset_r.csv'
    data = pd.read_csv(file_name)
    data.dropna(axis=0, inplace=True)
    mask = qf.remove_outliers(data, 0.1, 0.95)
    mask = mask.values & data['conf'].between(0.75, 2)

    # Normalizing data

    # Fixed parameters
    a_lim = 720  # limit (in pixels) for "a" direction
    # limit (in pixels) for "b" direction (dived by 2 to allow for negative direction. Origin is at center)
    b_lim = 1280 / 2

    cam_lat = data.iloc[0, 5]
    cam_lon = data.iloc[0, 6]

    scale_factor = 1000  # used to scale the delta lat-lon data to expose enough variability

    # Unpacking data
    a = data.iloc[:, 0].values[:, np.newaxis]
    b = data.iloc[:, 1].values[:, np.newaxis]
    c = data.iloc[:, 2].values[:, np.newaxis]
    gt_lat = data.iloc[:, 3].values[:, np.newaxis]
    gt_lon = data.iloc[:, 4].values[:, np.newaxis]

    delta_lat = gt_lat - cam_lat
    delta_lon = gt_lon - cam_lon

    # Normalizing data
    # Input
    a_norm = a / a_lim
    b_norm = b / b_lim

    # Output
    delta_lat_norm = delta_lat * scale_factor
    delta_lon_norm = delta_lon * scale_factor

    # Checking correlation
    fig1, axs1 = plt.subplots(2, 2)
    # 1st row, 1st column (a)
    axs1[0, 0].plot(a_norm[mask], 'o', label='data')
    axs1[0, 0].set(xlabel=None, ylabel="$a_{norm}$")
    # 2nd row, 1st column (b)
    axs1[1, 0].plot(b_norm[mask], 'o', label='data')
    axs1[1, 0].set(xlabel=None, ylabel="$b_{norm}$")
    # 1st row, 2nd column (c)
    axs1[0, 1].plot(delta_lat_norm[mask], 'o', label='data')
    axs1[0, 1].set(xlabel=None, ylabel="$delta_{lat_{norm}}$")
    # 2nd row, 2nd column (d)
    axs1[1, 1].plot(delta_lon_norm[mask], 'o', label='data')
    axs1[1, 1].set(xlabel=None, ylabel="$delta_{lon_{norm}}$")

    # new plot
    # fig3, axs3 = plt.subplots(2, 2)
    # # 1st row, 1st column (a)
    # axs3[0, 0].plot(a_norm, 'o', label='data')
    # axs3[0, 0].plot(a_norm[mask], '*', label='outliers removed')
    # axs3[0, 0].set(xlabel=None, ylabel="$a_{norm}$")
    # # 2nd row, 1st column (b)
    # axs3[1, 0].plot(b_norm, 'o', label='data')
    # axs3[1, 0].plot(b_norm[mask], '*', label='outliers removed')
    # fig3.show()

    # Training Quadratic fit
    X = np.array([a_norm[:], b_norm[:]])
    X = X.reshape(X.shape[0:2])
    y_lat = delta_lat_norm
    y_lon = delta_lon_norm

    qf_instance = qf('weights')

    q_lat_check, _, rank1, _ = qf.train_quad(X[:, mask], y_lat[mask])
    q_lon_check, _, rank2, _ = qf.train_quad(X[:, mask], y_lon[mask])
    qf_instance.train_weights(X, y_lat, y_lon)

    q_lat = qf_instance.q_lat
    q_lon = qf_instance.q_lon

    dir_name = 'weights'
    qf_instance.save_weights(dir_name)

    # Checking the fit on the training data
    n_sample = max(delta_lat_norm.shape)

    delta_lat_quad = np.zeros((n_sample, 1))
    delta_lon_quad = np.zeros((n_sample, 1))

    lat_quad = np.zeros((n_sample, 1))
    lon_quad = np.zeros((n_sample, 1))

    for sample in range(n_sample):
        if mask.values[sample]:
            x_sample = np.array([a_norm[sample], b_norm[sample]])  # input

            delta_lat_quad[sample] = qf.quad_fun(x_sample, q_lat)
            delta_lon_quad[sample] = qf.quad_fun(x_sample, q_lon)

            lat_quad[sample], lon_quad[sample] = qf_instance.quad_predict(a_norm[sample], b_norm[sample], cam_lat,
                                                             cam_lon, scale_factor)

    # edit 2nd column of subplot
    axs1[0, 1].plot(delta_lat_quad[mask], '*', label='quad')
    axs1[1, 1].plot(delta_lon_quad[mask], '*', label='quad')
    for i in axs1:
        for j in i:
            j.legend(loc="upper right")
    fig1.savefig('Figures/one')
    fig1.show()

    # create new subplot figure
    fig2, axs2 = plt.subplots(1, 2)
    axs2 = axs2[:, np.newaxis].T
    # 1st row, 1st column (a)
    axs2[0, 0].plot(gt_lat[mask], 'o', label='data')
    axs2[0, 0].plot(lat_quad[mask], '*', label='quad')
    # 1st row, 2nd column (b)
    axs2[0, 1].plot(gt_lon[mask], 'o', label='data')
    axs2[0, 1].plot(lon_quad[mask], '*', label='quad')
    for i in axs2:
        for j in i:
            j.legend(loc="upper right")
    fig2.savefig('Figures/two')
    fig2.show()
    print("Done")


if __name__ == '__main__':
    AI_track_sea_cam2loc()
    print('Done')
