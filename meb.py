import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# def linearKernel_dist (in1, in2):
#     dist = in1-in2
#     return np.dot(dist,np.transpose(dist))


def distance(x1, x2, y1, y2, idx1, idx2, C, kernel, gamma=None):
    dist = 0
    if(idx1 != idx2):
        if (kernel == 'linear'):
            dist_vec = y1*x1 - y2*x2
            dist = 2*(1+1/C-y1*y2) + np.dot(dist_vec, np.transpose(dist_vec))
        elif (kernel == 'rbf'):
            dist_vec = x1-x2
            dist = 2 * (2+1/C - y1*y2*(np.exp(-gamma*np.dot(dist_vec, np.transpose(dist_vec)))+1))
    else:
        dist = 0

    return dist

def inner_product(x1, x2, y1, y2, idx1, idx2, C, kernel, gamma=None):
    prod = 0
    delta = 0
    if(idx1 != idx2):
        delta = 0
    else:
        delta = 1

    if (kernel == 'linear'):
        prod = y1*y2*np.dot(x1, np.transpose(x2))+y1*y2+delta/C
    elif (kernel == 'rbf'):
        prod = y1*y2*np.exp(-gamma * np.dot(x1, np.transpose(x2)))+y1*y2+delta/C

    return prod


def MEB (core_set, y_core, core_size, epsilon, C, kernel, gamma=None):

    first_index = np.random.randint(0, core_size, 1)[0]

    dist_list = np.zeros(core_size)

    bias_term = np.zeros(core_size)

    dist_max = 0

    p = core_set[first_index]
    py = y_core[first_index]

    bias_term[first_index] = bias_term[first_index] + 1/C

    far_list = []
    far_y_list = []
    far_index_list = []
    far_list.append(p)
    far_y_list.append(py)
    far_index_list.append(first_index)

    kernel_contant = np.ones(core_size) * (4+2/C)

    # initialize farthest point p away from the center c
    core_rbf = rbf_kernel(np.array(far_list), core_set, gamma) + 1
    core_rbf_sum = np.multiply(np.dot(np.array(far_y_list).reshape(-1), core_rbf), y_core) + bias_term
    dist_list = kernel_contant - core_rbf_sum * 2
    index = np.argmax(dist_list).reshape(-1)[0]


    p = core_set[index]
    py = y_core[index]

    bias_term[index] = bias_term[index] + 1 / C

    far_list.append(p)
    far_y_list.append(py)
    far_index_list.append(index)

    # iteration for constructing the core-set
    for cnt in range(int(2/epsilon)):
        core_rbf = np.multiply(rbf_kernel([p], core_set, gamma) + 1, y_core) * py
        core_rbf_sum = core_rbf_sum + core_rbf[0]
        core_rbf_sum[index] = core_rbf_sum[index] + 1 / C
        dist_list = kernel_contant - core_rbf_sum * 2 / (cnt+1)
        index = np.argmax(dist_list).reshape(-1)[0]

        p = core_set[index]
        py = y_core[index]

        bias_term[index] = bias_term[index] + 1 / C

        far_list.append(p)
        far_y_list.append(py)
        far_index_list.append(index)

    dist_max = dist_list[index]

    far_index_list = np.unique(far_index_list)

    return dist_max, dist_list, far_index_list



if __name__ == "__main__":
    dim = 10
    theta = 0.1
    data_size = 1000
    inp = np.random.rand(data_size,dim)
    y_core = np.ones(data_size)

    inp_re = np.random.rand(0,dim)
    y_re = np.ones(0)

    # R, dist_re, dist_core = MEB(inp, y_core, inp_re, y_re, data_size, 0, theta, 1.0, 'linear')
    # R, dist_re, dist_core = MEB(inp, y_core, inp_re, y_re, data_size, 0, theta, 1.0, 'rbf', 0.5)
    #
    # print('Smallest R is '+ str(R))
    # print ('dist_core is ' + str(dist_core))
    # print ('dist_re is ' + str(len(dist_re)))
