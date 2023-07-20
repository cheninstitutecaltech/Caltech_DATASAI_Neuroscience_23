''' This code is adapted from search_util.py in the fges-py Github respository
by Iman Wahle on 7/9/2023. '''

import numpy as np


def mean_shift_data(data):
    '''Shift all variables in a dataset to have mean zero'''
    return data - np.mean(data, axis=0)

def estimate_parameters(dag, data):
    '''
    Estimate the parameters of a DAG to fit the data.
    :return: matrix of edge coefficients, and diagonal matrix of residuals
    For the parameters matrix, p[i, j] is the weight of edge i -> j
    '''

    # assert get_undir_edge(dag) is None

    data = mean_shift_data(data)
    num_nodes = dag.shape[0]

    edge_parameters = np.zeros((num_nodes, num_nodes))
    residuals = np.zeros((num_nodes, num_nodes))

    for j in range(num_nodes):
        inbound_nodes = [i for i in range(num_nodes) if dag[j,i]==1]

        if len(inbound_nodes) == 0:
            residuals[j, j] = np.var(data[:, j])
            continue

        assert j not in inbound_nodes

        a = data[:, inbound_nodes]
        b = data[:, j]

        params, r, _, _ = np.linalg.lstsq(a, b, rcond=None)

        residuals[j, j] = r / (data.shape[0] - 1)

        for i in range(len(inbound_nodes)):
            edge_parameters[inbound_nodes[i], j] = params[i]
            # v = edge_parameters * v + e

    return np.array(edge_parameters), np.array(residuals)


def get_covariance_matrix(params, resids):
    '''
    Get the covariance matrix from edge parameters
     (representing a DAG) and the residuals.

    For the equation, see "Causal Mapping of Emotion Networks in the Human Brain" (p. 15)
    The params matrix is taken with orientation p[i, j] is the weight for edge i -> j
    '''
    id = np.identity(params.shape[0])
    a = np.linalg.inv(id - params.transpose())

    return np.matmul(np.matmul(a, resids), np.transpose(a))

def get_correlation_matrix(params, resids):
    '''
    Get the correlation matrix from edge parameters
     (representing a DAG) and the residuals.

    For the covariance equation, see "Causal Mapping of Emotion Networks in the Human Brain" (p. 15)
    The params matrix is taken with orientation p[i, j] is the weight for edge i -> j

    Each entry in the covariance matrix is normalized by \sigma_i*\sigma_j
    to get the correlation matrix.
    '''

    cov = get_covariance_matrix(params, resids)
    stdistdj = np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    corr = cov / stdistdj
    return corr