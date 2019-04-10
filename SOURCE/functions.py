import pandas as pd
import numpy as np
import SOURCE.IndoorLocKNN as IPS_knn
import SOURCE.IndoorLocQL as IPS_ql
import SOURCE.IndoorLocDRL as IPS_drl


# -----------------------------------------------------------
# -----------------------------------------------------------
# Read a csv file and return the data as numpy matrix
def get_data_with_header(filename):
    fp = pd.read_csv(filename)
    return fp.values


# -----------------------------------------------------------
# -----------------------------------------------------------
# Read a csv file and return the data as numpy matrix
def get_data_without_header(filename):
    fp = pd.read_csv(filename, header=None)
    return fp.values


# -----------------------------------------------------------
# -----------------------------------------------------------
# Save a csv file to disk
def save_data(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename,  index=False, header=False, float_format='%.2f')


# -----------------------------------------------------------
# get_stats_aps
# -----------------------------------------------------------
# Return a np array of dim n_aps x 3
# [0] -> ap id
# [1] -> pct of samples with data (!=100)
# [2] -> pct of samples with rssi > TH1  (strong signal, i.e. -50)
def get_stats_aps(fps, th1):
    n_samples = fps.shape[0]   # number of rows == number of samples
    n_aps = fps.shape[1]       # number of columns == number of APS
    stats_aps = np.zeros([n_aps, 3])

    idx_no_100 = fps != 100
    idx_th1 = np.logical_and(fps != 100, fps > th1)

    stats_aps[:, 0] = range(n_aps)
    stats_aps[:, 1] = sum(idx_no_100)/n_samples
    stats_aps[:, 2] = sum(idx_th1)/n_samples

    return stats_aps


# -----------------------------------------------------------
# -----------------------------------------------------------
def get_only_good_aps(fps, TH_PCT_GOOD_AP, TH_RSSI_GOOD):
    stats_aps = get_stats_aps(fps, TH_RSSI_GOOD)
    n_aps = fps.shape[1]

    v_good_aps = np.zeros(n_aps, dtype=bool)
    for i in range(stats_aps.shape[0]):
        if stats_aps[i, 2] >= TH_PCT_GOOD_AP:
            v_good_aps[i] = True

    new_fps = fps[:, v_good_aps]
    return new_fps


# -----------------------------------------------------------
# normalize01
# -----------------------------------------------------------
# 100 -> 0
# -x -> between 0 and 1
# 0 means no data or very poor signal.
# 1 means very strong signal
def normalize_rssi_01(data, min_value):
    index_100 = data == 100
    new_data = (data + abs(min_value)) / abs(min_value)
    new_data[index_100] = 0
    return new_data


# -----------------------------------------------------------
# -----------------------------------------------------------
# Normalize RSSI data between 0 and 1
def normalize_rssi(train_fps, test_fps):
    min_rssi = np.amin(train_fps)
    train_fps_norm = normalize_rssi_01(train_fps, min_rssi)
    test_fps_norm = normalize_rssi_01(test_fps, min_rssi)
    return train_fps_norm, test_fps_norm


# -----------------------------------------------------------
# -----------------------------------------------------------
def normalize_xy(train, test):
    min_v = np.amin(train)
    max_v = np.amax(train)

    train_norm = (train - min_v) / (max_v - min_v)
    test_norm = (test - min_v) / (max_v - min_v)
    return train_norm, test_norm


# -----------------------------------------------------------
# -----------------------------------------------------------
# run knn algorithm
def ips_knn(train_data, test_data, train_loc, test_loc, k, conf, floor_estimation, building_estimation):
    results_name = "../RESULTS/knn." + conf.get_experiment_name() + ".csv"

    indoorloc_model = IPS_knn.IndoorLocKNN(train_data, train_loc, k, floor_estimation, building_estimation)
    estimated_loc, v_error, mean_acc, p75_acc = indoorloc_model.get_accuracy(test_data, test_loc)

    debug_data = np.concatenate((test_loc,
                                 estimated_loc,
                                 np.reshape(v_error, [len(v_error), 1])), axis=1)
    save_data(debug_data, results_name)

    return mean_acc, p75_acc


# -----------------------------------------------------------
# -----------------------------------------------------------
# run ql algorithm
def ips_ql(train_data, test_data, train_loc, test_loc, do_training, do_grid_mode, conf):
    model_name = "../MODELS/ql." + conf.get_experiment_name()
    results_name = "../RESULTS/ql." + conf.get_experiment_name() + ".csv"

    indoorloc_model = IPS_ql.IndoorLocQL(train_data, train_loc, conf, do_training, do_grid_mode, model_name) #Train
    estimated_loc, v_error, mean_acc, p75_acc, est_results = indoorloc_model.get_accuracy(test_data, test_loc) #Test and return accuracy
    v_error_goods = v_error[np.array(est_results, dtype=bool)]

    debug_data = np.concatenate((test_loc,
                                 estimated_loc,
                                 np.reshape(v_error, [len(v_error), 1]),
                                 np.reshape(est_results, [len(est_results), 1])), axis=1)
    save_data(debug_data, results_name)

    return mean_acc, p75_acc, np.sum(est_results) / len(est_results), np.mean(v_error_goods), np.percentile(v_error_goods, 75)


# -----------------------------------------------------------
# -----------------------------------------------------------
# run drl algorithm
def ips_drl(train_data, test_data, train_loc, test_loc, do_training, do_grid_mode, conf):
    model_name = "../MODELS/drl." + conf.get_experiment_name()
    results_name = "../RESULTS/drl." + conf.get_experiment_name() + ".csv"

    print("------------")
    print("- Training -")
    print("------------")

    indoorloc_model = IPS_drl.IndoorLocDRL(train_data, train_loc, conf, do_training, do_grid_mode, model_name) #Train

    print("------------")
    print("- Testing  -")
    print("------------")

    estimated_loc, v_error, mean_acc, p75_acc, est_results = indoorloc_model.get_accuracy(test_data, test_loc) #Test and return accuracy
    v_error_goods = v_error[np.array(est_results, dtype=bool)]

    debug_data = np.concatenate((test_loc,
                                 estimated_loc,
                                 np.reshape(v_error, [len(v_error), 1]),
                                 np.reshape(est_results, [len(est_results), 1])), axis=1)
    save_data(debug_data, results_name)

    return mean_acc, p75_acc, np.sum(est_results) / len(est_results), np.mean(v_error_goods), np.percentile(v_error_goods, 75)
