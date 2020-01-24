from Data_dist import data_partition
from meb import MEB
import numpy as np
from sklearn.svm import SVC
from Data_load import read_UCI_data
import sys, getopt
import pandas as pd

def main(argv):
    edge_node_n = 5
    C = 7.0
    gamma = 0.5
    dataset = 1
    data_part = 0
    n_part = 5
    equal_flag = True

    try:
        opts, args = getopt.getopt(argv, 'hn:p:d:t:e:',['help','Edge_node_n=','data_part=','dataset=','n_part=','equal_flag='])
    except getopt.GetoptError:
        print('Distributed_core_set.py -n <Edge_node_n> -p <data_part> -d <dataset> -t <n_part> -e <equal_flag>')
        sys.exit(2)

    for opt, arg in opts:
        if (opt == '-h'):
            print('Distributed_core_set.py -n <Edge_node_n> -p <data_part> -d <dataset> -t <n_part> -e <equal_flag>')
            print('-p 0: data_partition; 1: kmeans_partition; 2: kmeans_random')
            print('-d 0: skin_nonskin; 1: phishing; 2: NB15; 3: ijcnn; 4: covtype')
            print('-e True: equal; False: not equal')
            sys.exit()
        elif (opt == '-n'):
            edge_node_n = int(arg)
        # elif (opt == '-C'):
        #     C = float(arg)
        # elif (opt == '-g'):
        #     gamma = float(arg)
        elif (opt == '-p'):
            data_part = int(arg)
        elif (opt == '-d'):
            dataset = int(arg)
        elif (opt == '-t'):
            n_part = int(arg)
        elif (opt == '-e'):
            equal_flag = arg

    if(dataset == 0):
        C_set = np.arange(1.0, 18.0, 2.0)
        gamma_set = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
    elif(dataset == 1):
        C_set = np.arange(1.0, 11.0, 1.0)
        gamma_set = np.array([0.01, 0.03, 0.1, 0.3, 0.5, 1.0])
    elif (dataset == 3):
        C_set = np.arange(1.0, 16.0, 2.0)
        gamma_set = np.array([0.1, 0.3, 0.5, 1.0, 2.0])


    epsilon = 0.001

    # load training data
    train_data, train_label, test_data, test_label = read_UCI_data(dataset)

    # separate data
    edge_data, edge_label, global_index = data_partition(train_data, train_label, edge_node_n)

    C_list = []
    gamma_list = []
    upload_n_list = []
    SV_n_list = []
    accuracy_list = []

    for C in C_set:
        for gamma in gamma_set:

            C_list.append(C)
            gamma_list.append(gamma)

            # MEB at each edge node
            upload_n = 0
            global_set = []
            for node in range(edge_node_n):
                train_size = np.size(edge_label[node])
                R, dist_core, core_set = MEB(edge_data[node], edge_label[node], train_size, epsilon, C, 'rbf',
                                             gamma=gamma)
                upload_n = upload_n + np.size(core_set)
                global_set.append(global_index[node][core_set])
            # print(upload_n)
            upload_n_list.append(upload_n)

            core_set = global_set[0]
            for node in range(1, edge_node_n):
                core_set = np.append(core_set, global_set[node])

            core_train = train_data[core_set]
            core_label = train_label[core_set]

            core_model = SVC(C=C, kernel='rbf', gamma=gamma)
            core_model.fit(core_train, core_label)
            core_SV_size = core_model.n_support_[0] + core_model.n_support_[1]

            SV_n_list.append(core_SV_size)
            accuracy_list.append(core_model.score(test_data, test_label))

            # print(core_SV_size)
            # print(core_model.score(test_data, test_label))


    dataframe = pd.DataFrame({'C': C_list, 'gamma': gamma_list, '# of upload': upload_n_list, '# of SV': SV_n_list, 'test_accuracy': accuracy_list})

    if (dataset == 0):
        file_name = 'skin_nonskin_' + str(edge_node_n) + '_nodes.csv'
    elif (dataset == 1):
        file_name = 'phishing_' + str(edge_node_n) + '_nodes.csv'
    elif (dataset == 2):
        file_name = 'NB15_' + str(edge_node_n) + '_nodes.csv'
    elif (dataset == 3):
        file_name = 'ijcnn1_' + str(edge_node_n) + '_nodes.csv'
    elif (dataset == 4):
        file_name = 'covtype_' + str(edge_node_n) + '_nodes.csv'
    else:
        raise ValueError
    file_name = 'results/'+file_name

    dataframe.to_csv(file_name, index=False, sep=',')

if __name__ == "__main__":
    main(sys.argv[1:])


