"""
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
    Student code version with parts omitted.
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import pdb
from scipy.stats import norm

NUM_PARTICLES = 1000

# define the randomness in theta & distance for propagating state
STDDEV_THETA =  0.0872665 # radians, 5 degrees
STDDEV_DISTANCE = 0.1 # meters
# stddev for defining particle weight in correction step
STDDEV_MEAS_ERR = 0.5 # meters

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters
DELTA_T = 0.1  # assume "10^5" uS = 0.1 s


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin) * \
        math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propogate_state(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    state vector is [x_t; y_t; theta_t] all in the global frame

    """
    x_prev = x_t_prev[0]
    y_prev = x_t_prev[1]
    theta_prev = x_t_prev[2]

    # sample random theta & distance
    samp_theta = theta_prev + np.random.normal(0, STDDEV_THETA)
    samp_dist = np.random.normal(0, stddev_dist)

    # sample predicted state x_t_i with probability P(x_t_i | x_t-1_i , u_t)
    x_t = x_prev + samp_dist * np.cos(samp_theta)
    y_t = y_prev + samp_dist * np.sin(samp_theta)
    theta_t = samp_theta
    theta_t = wrap_to_pi(theta_t)

    x_bar_t = np.array([x_t, y_t, theta_t])

    return x_bar_t





def prediction_step(P_t_prev, u_t):
    """Compute the prediction of EKF

    Parameters:
    P_t_prev (np.array)         -- the previous state matrix, [[x, y, theta, w]^T [x, y, theta, w]^T ...]
    u_t (np.array)              -- the control input

    Returns:
    P_t_predict (np.array)      -- the predicted state matrix after propagating all particles forward in time
    """

    # Iterate through every column (particle) of the state matrix and propagate them forward.

    for i in range(NUM_PARTICLES):
        P_t_predict(:2,i) = propogate_state(P_t_prev(:2,i), u_t) #at this point the weight stays zero

    return P_t_predict

def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    z_bar_t defined as [z_xLL, z_yLL, z_theta]
    """

    """STUDENT CODE START"""
    x_t = x_bar_t[0]
    y_t = x_bar_t[1]
    theta_t = x_bar_t[2]
    delta_x = X_LANDMARK - x_t
    delta_y = Y_LANDMARK - y_t

    # Tim has the derivation for this
    z_xLL = np.sin(theta_t) * delta_x - np.cos(theta_t) * delta_y
    z_yLL = np.cos(theta_t) * delta_x + np.sin(theta_t) * delta_y

    # z_xLL = np.cos(theta_t) * delta_x - np.sin(theta_t) * delta_y
    # z_yLL = np.sin(theta_t) * delta_x + np.cos(theta_t) * delta_y
    z_theta = theta_t

    z_bar_t = np.array([z_xLL, z_yLL, z_theta])
    """STUDENT CODE END"""

    return z_bar_t


def correction_step(P_t_predict, z_t):
    """Compute the correction of EKF

    Parameters:
    P_t_predict         (np.array)    -- the predicted state matrix time t
    z_t                 (np.array)    -- the measured state of time t

    Returns:
    P_t                 (np.array)    -- the final state matrix estimate of time t
    """
    normal_object = scipy.stats.norm(0, STDDEV_MEAS_ERR)

    # Calculate weight for each particle j
    for j in range(NUM_PARTICLES):
        # calculate what the particle thinks the measurement should be 
        z_bar_t = calc_meas_prediction(x_bar_t) # TODO: CHANGE THIS FUNCTION

        # find the geometric distance between z_t and z_pred (Pythagorean theorem)
        distance = np.linalg.norm(z_t - z_bar_t)

        # weight is zero-mean normal PDF evaluated at d (what variance?)
        weight = normal_object.pdf(distance)

        P_t_predict(3,j) = weight

    # Sample from the set of particles proportional to their weights
    # draw another card from the deck of particle-cards with probability w
    P_t = np.random.choice(P_t_prev(:2,:).to_list(), size=(3,NUM_PARTICLES), replace=True, p=P_t_prev(3,:))     
    # TODO: FINISH THIS!
    # np.random.choice(range(NUM_PARTICLES), size=(NUM_PARTICLES), replace=True, p=P_t_prev[3,:])

    return P_t


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = "./logs/"
    filename = "2020_2_26__16_59_7"
    # filename = "2020_2_26__17_21_59"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    # Tim changed yaw to radians
    yaw_lidar = [wrap_to_pi(x * -np.pi / 180) for x in data["Yaw"]]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    """STUDENT CODE START"""
    N = 7  # number of states
    # assume it starts at the origin
    state_est_t_prev = np.array([0, 0, yaw_lidar[0], 0, 0, 0, yaw_lidar[0]])
    var_est_t_prev = np.identity(N)

    state_estimates = np.empty((N, len(time_stamps)))
    covariance_estimates = np.empty((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    """STUDENT CODE END"""
    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        """STUDENT CODE START"""

        u_t = np.array([x_ddot[t], y_ddot[t]]  # since x_ddot is
                       )  # u_t = [a_x_t, a_y_t] is a 2x815 array.
        # a_x_t - acceleration in x
        # a_y_t - acceleration in y
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(
            state_est_t_prev, u_t, var_est_t_prev)

        # Get measurement
        """STUDENT CODE START"""
        z_t = np.array([x_lidar[t], y_lidar[t], yaw_lidar[t]])
        """STUDENT CODE END"""

        # Correction Step
        state_est_t, var_est_t, z_bar_t = correction_step(state_pred_t,
                                                          z_t,
                                                          var_pred_t)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data

        state_estimates[:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

        """STUDENT CODE START"""
        # Plot or print results here
        if np.mod(t, 5) == 0:
            plt.subplot(2, 1, 1)
            plt.plot(gps_estimates[0, t],
                     gps_estimates[1, t], 'b.', label='GPS')

            plt.quiver(state_estimates[0, t], state_estimates[1, t], np.cos(
                state_estimates[2, t]), np.sin(state_estimates[2, t]), color='r')

            plt.quiver(state_estimates[0, t], state_estimates[1, t], np.cos(
                z_t[2]), np.sin(z_t[2]), color='g')

            plt.xlim(-4, 14)
            plt.ylim(-14, 4)
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            if t == 0:
                plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(z_bar_t[0], z_bar_t[1], 'r.', label='Estimated Z')
            plt.plot(z_t[0], z_t[1], 'k.', label='Actual Z')

            plt.xlabel('Right (m)')
            plt.ylabel('Forward (m)')
            if t == 0:
                plt.legend()

            print(t)
            plt.pause(0.01)

            # if t == 450:
            #    pdb.set_trace()
            # At t=450, the compass is facing left (West) but it doesn't seem to think it's to the south of the landmark.
        """STUDENT CODE END"""

    plt.subplot(2, 1, 1)
    plt.plot(gps_estimates[0, :], gps_estimates[1, :], 'b.', label='GPS')

    plt.plot(state_estimates[0, :], state_estimates[1,
                                                    :], color='r', label='State Estimate')

    plt.xlim(-4, 14)
    plt.ylim(-14, 4)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(z_bar_t[0], z_bar_t[1], 'r.', label='Estimated Z')
    plt.plot(z_t[0], z_t[1], 'k.', label='Actual Z')

    plt.xlabel('Right (m)')
    plt.ylabel('Forward (m)')
    plt.show()
    return 0


if __name__ == "__main__":
    main()
