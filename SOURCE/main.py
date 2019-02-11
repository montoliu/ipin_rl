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

    training_filename = conf.TRAINING_FILENAME
    database = conf.DATABASE

    # open datasets
    # if DO_SETS == true: open and create the datasets
    # if DO_SETS == false: open previously created datasets
    if conf.DO_SETS:
        fp = myf.get_data_with_header(database)
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

        if conf.DO_ONLY_GOOG_APS:
            myf.save_data(x_train, "../TMP/x_train_good.csv")
            myf.save_data(x_test, "../TMP/x_test_good.csv")
        else:
            myf.save_data(x_train, "../TMP/x_train.csv")
            myf.save_data(x_test, "../TMP/x_test.csv")

        myf.save_data(y_train, "../TMP/y_train.csv")
        myf.save_data(y_test, "../TMP/y_test.csv")
    else:
        if conf.DO_ONLY_GOOG_APS:
            x_train = myf.get_data_without_header("../TMP/x_train_good.csv")
            x_test = myf.get_data_without_header("../TMP/x_test_good.csv")
        else:
            x_train = myf.get_data_without_header("../TMP/x_train.csv")
            x_test = myf.get_data_without_header("../TMP/x_test.csv")

        y_train = myf.get_data_without_header("../TMP/y_train.csv")
        y_test = myf.get_data_without_header("../TMP/y_test.csv")

    print("Train data has: " + str(x_train.shape[0]) + " samples")
    print("Test data has : " + str(x_test.shape[0]) + " samples")

    # execute the algorithms and show the results
    if conf.DO_KNN:
        print("\n--------------------- ")
        print("--- KNN Algorithm --- ")
        print("--------------------- ")
        knn_p50, knn_p75 = myf.ips_knn(x_train, x_test, y_train, y_test, conf.KNN_K,
                                       floor_estimation=False, building_estimation=False)
        print("Mean: " + "{:0.2f}".format(knn_p50) + ", 75% percentile: " + "{:0.2f}".format(knn_p75))
        print("--------------------- ")

    if conf.DO_DRL:
        print("\n--------------------- ")
        print("--- DRL Algorithm --- ")
        print("--------------------- ")
        drl_p50, drl_p75 = myf.ips_drl(x_train, x_test, y_train, y_test, conf.DO_DRL_TRAINING, conf.DO_GRID_MODE,
                                       training_filename, conf)
        print("DRL -> Mean: " + "{:0.2f}".format(drl_p50) + ", 75% percentile: " + "{:0.2f}".format(drl_p75))
        print("--------------------- ")

    if conf.DO_QL:
        print("\n--------------------- ")
        print("--- QL  Algorithm --- ")
        print("--------------------- ")
        ql_p50, ql_p75 = myf.ips_ql(x_train, x_test, y_train, y_test, conf.DO_QL_TRAINING, conf.DO_GRID_MODE,
                                   training_filename, conf)
        print("Mean: " + "{:0.2f}".format(ql_p50) + ", 75% percentile: " + "{:0.2f}".format(ql_p75))
        print("--------------------- ")


