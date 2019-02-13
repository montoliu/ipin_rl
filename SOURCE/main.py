import SOURCE.functions as myf
import SOURCE.configuration as config
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
# LOG 0 Main messages
# LOG 1 DRL messages
# LOG 2 NN messages
if __name__ == "__main__":
    # lead configuration variables
    conf = config.configuration()

    # Names of the train and test files
    if conf.DO_ONLY_GOOG_APS:
        x_train_name = "../TMP/" + conf.EXPERIMENT_NAME + ".x_train_good.csv"
        x_test_name = "../TMP/" + conf.EXPERIMENT_NAME + ".x_test_good.csv"
    else:
        x_train_name = "../TMP/" + conf.EXPERIMENT_NAME + ".x_train.csv"
        x_test_name = "../TMP/" + conf.EXPERIMENT_NAME + ".x_test.csv"

    y_train_name = "../TMP/" + conf.EXPERIMENT_NAME + ".y_train.csv"
    y_test_name = "../TMP/" + conf.EXPERIMENT_NAME + ".y_test.csv"

    # open datasets
    # if DO_SETS == true: open and create the datasets
    # if DO_SETS == false: open previously created datasets
    if conf.DO_SETS:
        print("Creating train and test: ")
        print("TRAIN: " + x_train_name + " " + y_train_name)
        print("TEST:  " + x_test_name + " " + y_test_name)
        print("")

        fp = myf.get_data_with_header(conf.DATABASE)
        NUM_WAPS = fp.shape[1] - 9
        n_samples = fp.shape[0]
        x_data = fp[:, 0:NUM_WAPS]
        y_data = fp[:, NUM_WAPS:NUM_WAPS + 2]  # get only x and y
        print("The global dataset has " + str(n_samples) + " samples and " + str(NUM_WAPS) + " WAPS")

        if conf.DO_ONLY_GOOG_APS:
            x_data = myf.get_only_good_aps(x_data, conf.TH_PCT_GOOD_AP, conf.TH_RSSI_GOOD)
            NUM_WAPS = x_data.shape[1]
            print("After reducing to only goods WAPS, there are " + str(NUM_WAPS) + " good WAPS")

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
        x_train, x_test = myf.normalize_rssi(x_train, x_test)

        myf.save_data(x_train, x_train_name)
        myf.save_data(x_test, x_test_name)
        myf.save_data(y_train, y_train_name)
        myf.save_data(y_test, y_test_name)
    else:
        print("Reading train and test: ")
        print("TRAIN: " + x_train_name + " " + y_train_name)
        print("TEST:  " + x_test_name + " " + y_test_name)
        print("")

        x_train = myf.get_data_without_header(x_train_name)
        x_test = myf.get_data_without_header(x_test_name)
        y_train = myf.get_data_without_header(y_train_name)
        y_test = myf.get_data_without_header(y_test_name)

    print("Train data has: " + str(x_train.shape[0]) + " samples")
    print("Test data has : " + str(x_test.shape[0]) + " samples")
    print("There are " + str(x_train.shape[1]) + " WAPS")

    # execute the algorithms and show the results
    if conf.DO_KNN:
        print("\n--------------------- ")
        print("--- KNN Algorithm --- ")
        print("--------------------- ")
        knn_p50, knn_p75 = myf.ips_knn(x_train, x_test, y_train, y_test, conf.KNN_K, conf, floor_estimation=False, building_estimation=False)
        print("--------------------- ")
        print("Mean: " + "{:0.2f}".format(knn_p50) + ", 75% percentile: " + "{:0.2f}".format(knn_p75))
        print("--------------------- ")

    if conf.DO_DRL:
        print("\n--------------------- ")
        print("--- DRL Algorithm --- ")
        print("--------------------- ")
        drl_p50, drl_p75 = myf.ips_drl(x_train, x_test, y_train, y_test, conf.DO_DRL_TRAINING, conf.DO_GRID_MODE, conf)
        print("--------------------- ")
        print("DRL -> Mean: " + "{:0.2f}".format(drl_p50) +
              ", 75% percentile: " + "{:0.2f}".format(drl_p75))
        print("--------------------- ")

    if conf.DO_QL:
        print("\n--------------------- ")
        print("--- QL  Algorithm --- ")
        print("--------------------- ")
        ql_p50, ql_p75, ql_pct = myf.ips_ql(x_train, x_test, y_train, y_test, conf.DO_QL_TRAINING, conf.DO_GRID_MODE, conf)
        print("--------------------- ")
        print("Mean: " + "{:0.2f}".format(ql_p50) +
              ", 75% percentile: " + "{:0.2f}".format(ql_p75) +
              ", Pct " + "{:0.2f}".format(ql_pct))
        print("--------------------- ")


