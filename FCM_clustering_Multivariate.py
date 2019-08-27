#!/usr/bin/env python
# coding: utf-8

# In[67]:


# %%bash
# "--user" is essential to install in local environment"
# pip install -U scikit-fuzzy


# In[68]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# from __future__ import division  # floating point division
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from scipy import optimize
from tkinter import *
import time
import load_data as dtl


# In[69]:


a = [x for x in range(10,20)]
print(a)


# In[70]:


def data_generator(num_cntr, dim, num_data):
    colors = ['y', 'g', 'b', 'orange']
#     num_cntr = 4
#     dim = 3
    centers = np.ndarray(shape=(num_cntr, dim), dtype=float)
    sigmas = np.ndarray(shape=(num_cntr, dim), dtype=float)

    for i in range(num_cntr):
        for j in range(dim):
            centers.itemset((i,j), np.random.uniform(0, 10))
            sigmas.itemset((i,j), np.random.uniform(0, 1.5))
#     centers = [[2, 2, 4],
#                [6, 5, 6],
#                [4, 4, 4],
#                [5,7, 9]]
#     sigmas = [[0.8, 0.3, 0.5],
#               [0.3, 0.5, 0.7],
#               [1.1, 0.7, 1.0],
#               [0.2, 0.5, 1.0]]
    features = []
    for i in range(dim):
        im_data = np.zeros(1)
        for j in range(num_cntr):
            im_data = np.hstack((im_data, np.random.normal(centers[j][i],
                                                           sigmas[j][i], num_data)))
        im_data = np.delete(im_data, 0)
        features.append(im_data)
#     print(features, len(features[0]))
    all_data = np.zeros(shape = (1, len(features[0])))
    for i in range(dim):
        all_data = np.vstack((all_data, features[i]))
    all_data = np.delete(all_data, (0), axis=0)
#     print(all_data)

    return all_data, colors


# In[71]:


"""
Modification of the traditional fuzzy c mean source code
cited from https://pythonhosted.org/scikit-fuzzy/_modules/skfuzzy/cluster/_cmeans.html#cmeans
"""
def _cmeans0(data, u_old, c, m, var, method):
    """
    Single step in generic fuzzy c-means clustering algorithm.

    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.

    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
#     print('data defore transpose is ', data)
    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)
    if(method == 1):
        d = _distance_weighted_euclidean(data, cntr, var)
    elif(method == 2):
        d = _distance_Chebyshev(data, cntr, var)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum() #objective function records

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def _distance_weighted_euclidean(data, centers, var):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    var : 1d array, size(1, dimension of data)
        Variance of each data dimension

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    weight_euclidean = np.ndarray(shape=(len(centers), len(data)), dtype=float)
#     print('the weight euclidean ', weight_euclidean.shape)
    for i in range(len(centers)):
#         print('center[i] is ', centers[i])
#         print('data is ',data)
        im_data = data - centers[i]
#         print('im_data is ', im_data) #x^power /  var
#         im_data = np.power(im_data, 2)/var.T
#         print('0 index',im_data[0][2])
        im_data = np.power(im_data, 2)
        im_data = im_data/var
#         print(im_data.shape)

#         print('var is ', var)
#         print('after im_data is ', im_data)
        im_data = np.sum(im_data, axis=1)
#         print('final data is ', im_data)
#         im_data = np.sqrt(im_data)
        for j in range(len(data)):
            weight_euclidean.itemset((i,j), im_data[j])
#     print('the new method dist is ', weight_euclidean)
    return weight_euclidean

def _distance_Chebyshev(data, centers, var):
    """
    Euclidean distance from each point to each cluster center.

    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.

    var : 1d array, size(1, dimension of data)
        Variance of each data dimension

    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    weight_euclidean = np.ndarray(shape=(len(centers), len(data)), dtype=float)
#     print('the weight euclidean ', weight_euclidean.shape)
    for i in range(len(centers)):
#         print('center[i] is ', centers[i])
#         print('data is ',data)
        im_data = abs(data - centers[i])
#         print('im_data is ', im_data) #x^power /  var
#         im_data = np.power(im_data, 2)/var
#         print('var is ', var)
        im_data = np.max(im_data, axis=1)
#         print('final data is ', im_data)
#         im_data = np.sqrt(im_data)
        for j in range(len(data)):
            weight_euclidean.itemset((i,j), im_data[j])
#     print('the new method dist is ', weight_euclidean)
    return weight_euclidean


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.

    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.

    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans(data, c, m, error, maxiter, var, method, init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.

    var : 1d array, size(1, dimension of data)
        Variance of each data dimension

    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.

    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.

    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.

    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.

    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0
    Q = []

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m, var, method)
        jm = np.hstack((jm, Jjm))
        p += 1
        d1 = np.power(d,2)
        Q_value_matrix = np.multiply(np.power(u, 2), d1)
#         print(Q_value_matrix.shape)
        Q_value_matrix = np.sum(Q_value_matrix, axis=1)
        Q_value = np.sum(Q_value_matrix, axis=0)
#         print(Q_value )
        Q.append(Q_value)

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc, Q


# In[72]:


def membership_graph(ncenter, u, xpts, ypts):
#plotting the membership function value of each cluster by measuring the L2-norm
    fig = plt.figure(figsize=plt.figaspect(1/ncenter))
    # set up the axes for the first plot
    for i in range(ncenter):
        ax = fig.add_subplot(1, ncenter, i+1, projection='3d')

        # plot a 3D surface like in the example mplot3d/surface3d_demo
        [gx, gy] = np.meshgrid(xpts, ypts, indexing = 'ij')
        surf = ax.plot_surface(gx, gy, np.reshape(u[i],(len(u[i]), 1)), color = colors[i])
        ax.set_zlim(0, 1)
        plt.gca().invert_xaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Membership')
    fig.suptitle('Membership value for clustering data in Centers = {0}; '.format(ncenters))

    return None

def find_base_coverage(p):
    def f(x, p):
        return (-1)*((1-x)**p)*x
    if (p==0):
        return 1
    minimum = optimize.fmin(f, 0, args=(p,))
    print("find the min", minimum, ' with beta ', p)
    return minimum[0]

def normalization(raw_data):
    #Feature scaling
#     print('raw data is ',raw_data)
    max_element = np.max(raw_data)
    min_element = np.min(raw_data)
#     print('max element ',max_element)
#     normal_data = (raw_data - min_element) / (max_element - min_element)
#     normal_data = raw_data/max_element
    if(max_element == 0 and min_element == 0):
        normal_data = np.divide(raw_data, 1)
    else:
        normal_data = np.divide(raw_data, max_element)
#     print('after normalizing ', normal_data)
    return normal_data

def info_gran(u, dist, ncenters, dim, power):
    print('info granular')
#     print(dist.shape, type(dist))
    for i in range(ncenters):
        ro = 0.0
        dr = 0.025
        max_v =0.0
#         d = dist[i]
        for j in range(int(1/dr)):
            cov = 0.0
#             print(d)
            for k in range(u[1].size):
                if(dist[i][k]< dim*ro):
                    cov += u[i][k]
            if(max_v <= cov*((1-ro)**power)):
                max_v = cov*(1-ro)**power
                ro_opt = ro
#                 ro += dr
#             else:
#                 break
#         radius_lst = np.hstack((radius_lst, ro_opt))
            ro += dr
        print('cluster ', i+1, ro_opt, max_v)


# In[73]:


def cntr_dist_cal(ncenters, cntr, var ):
    cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    for i in range(ncenters):
        for j in range(ncenters):
#             sub_dist = np.zeros(cntr.shape[1])
#             print("cntr dist")
#             print(cntr[i],cntr[j])
            sub_dist = np.power(cntr[i]-cntr[j], 2)
#             print(sub_dist)
#             print(np.sum(sub_dist))
            cntr_dist.itemset((i,j), np.sqrt(np.sum(sub_dist)))
#     print('cntr dist is', cntr_dist)
    return cntr_dist

def distance_euclidean(data, centers, var):

    weight_euclidean = np.ndarray(shape=(len(centers), len(data)), dtype=float)
    print(data.shape)
#     print('the weight euclidean ', weight_euclidean.shape)
    for i in range(len(centers)):

        im_data = data - centers[i]
#         print('data',data)
#         print('cntr', centers[i])
#         print(im_data)
        im_data = np.power(im_data, 2)
#         im_data = im_data/var
#         print(im_data.shape)

        im_data = np.sum(im_data, axis=1)
        im_data = np.sqrt(im_data)
        for j in range(len(data)):
            weight_euclidean.itemset((i,j), im_data[j])
#     print('the new method dist is ', weight_euclidean)
    return weight_euclidean

# def cntr_dist_cal(ncenters, cntr, var ):
#     cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
#     for i in range(ncenters):
#         for j in range(ncenters):
#             cntr_dist.itemset((i,j), np.linalg.norm(cntr[i]-cntr[j]))

#     return cntr_dist

def if_intersect(cntr_dist, radius_lst, ncenters, cntr):

    intersect_num = 0
    radius_sum = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    for i in range(ncenters):
        for j in range(ncenters):
            radius_sum.itemset((i,j), (radius_lst[i] + radius_lst[j]))
    cntr_radius_diff = cntr_dist - radius_sum
#     print('diff', cntr_radius_diff)
    for i in range(ncenters):
        for j in range(ncenters):
            if(i != j and cntr_radius_diff[i][j]<0):
#                 print('diff', cntr_radius_diff)
#                 print('intersecting between ', cntr[i], ' and ', cntr[j])
                intersect_num += 1
    if(intersect_num > 0):
        return True, intersect_num/2
    else:
        return False, intersect_num/2



def generate_new_coverage(u, d, power, ncenters, dim):
    radius_lst = np.zeros(1)
    cov_lst = np.zeros(1)
    num = u[1].size
    for i in range(ncenters):
        ro = 0.0
        dr = 0.025
        max_v =0.0
#         d = dist[i]
        for j in range(int(1/dr)):
            cov = 0.0
#             print(d)
            for k in range(u[1].size):
                if(d[i][k]< dim*ro):
                    cov += u[i][k]
            if(max_v <= cov*((1-ro)**power)):
                max_v = cov*((1-ro)**power)
                ro_opt = ro
#                 cov_sp = (cov/num)*((1-(cov/num))**power)
            ro += dr
        radius_lst = np.hstack((radius_lst, ro_opt))
        cov_lst = np.hstack((cov_lst, max_v/num))

    radius_lst = np.delete(radius_lst, 0)
    cov_lst = np.delete(cov_lst, 0)
    return radius_lst, cov_lst


def fin_best_beta(ncenters, cntr, u, d, power, radius_lst,dim, var):
    cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    step_size = 0.1
#     np.linalg.norm(sig_data - cntr_data)
    cntr_dist = cntr_dist_cal(ncenters, cntr, var)
    intersect_num_lst = []
    power_lst = []
#     print(if_intersect(cntr_dist, radius_lst, ncenters, cntr))
#     print(cntr)
#     print(cntr_dist)
    power_lst.append(power)
    check_intersect, intersect_num = if_intersect(cntr_dist, radius_lst, ncenters, cntr)
    intersect_num_lst.append(intersect_num)
    if(check_intersect):
        while(True):
            power = power + step_size
            radius_lst, a = generate_new_coverage(u, d, power, ncenters,dim)
            check_intersect, intersect_num = if_intersect(cntr_dist, radius_lst, ncenters, cntr)
            power_lst.append(power)
            intersect_num_lst.append(intersect_num)
            if(not check_intersect):
                break
    elif(not check_intersect):
        while(True):
            power = power - step_size
            if(power<0):
                power = power + step_size
                break
            radius_lst, a = generate_new_coverage(u, d, power, ncenters,dim)
            check_intersect, intersect_num = if_intersect(cntr_dist, radius_lst, ncenters, cntr)
            power_lst.append(power)
            intersect_num_lst.append(intersect_num)
            if(check_intersect):
                power = power + step_size
                break

    print('optimal power value is ', power)

    return power, power_lst, intersect_num_lst

def similarity(d, radius_lst, ncenters, dim):
    sim = np.zeros((ncenters, ncenters))

    for i in range(ncenters):
        for j in range(ncenters):
            if(not (i == j) and i<j):
                im = np.zeros(1)
                for t in range(d.shape[1]):
                    if(d[i, t] < radius_lst[i] and d[j, t] < radius_lst[j]):
                        sim_result = np.subtract(1 ,np.divide(np.absolute(np.subtract(d[i, t],d[j, t])), ncenters))
#                         print(sim_result)
                        im = np.hstack((im, sim_result))
#                 print(np.sum(im))
                sim.itemset((i,j), np.sum(im))
    sim = np.divide(sim, dim)
    print('sim is')
    print(sim)
    return sim

def check_membership(d, radius_lst, ncenters):
    mem = []
#     sub_mem = []

    for i in range(ncenters):
#         print(i)
        sub_mem = []
        for j in range(d.shape[1]):
            if(d[i,j] < radius_lst[i]):
                sub_mem.append(j)
        mem.append(sub_mem)
#         print(len(sub_mem))
    return mem



# In[74]:


def cluster_plot_normal_data(features, colors, power, ncenters, dim, fuzzy_coe):
    var = np.ndarray(dim)


#     all_data = np.zeros(shape = (1, num_data))
    for i in range(dim):
        features[i] = normalization(features[i])
#         print(features[i])
        avg = np.mean(features[i])
#         print(avg)
        variance = np.sum(np.power(features[i]-avg,2))/float(features[i].size-1)
#         print(variance)
        var.itemset(i, variance+0.000000001)
#         var.itemset(i, np.var(features[i])+0.000000001)
#     print(features)
#     var = np.sqrt(var)
    all_data = features

    print('var' , var)

    # Set up the loop and plot
#     fig, ax = plt.subplots(2, ncenters)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
#     fig, ax = plt.subplots(2, ncenters)
#     fig.set_size_inches(ncenters*4, ncenters, forward=True)


    fpcs = []
    radius_lst = np.zeros(1)
    radius_lst_ch = np.zeros(1)
#     print('hi',dim)
    cntr, u, u0, d, jm, p, fpc, Q = cmeans(all_data, ncenters, fuzzy_coe,
                                    error=0.0000000000005, maxiter=2000, var=var, method =1,init=None)
#     print('cntr at ',cntr)
#     print('return data', d)
#     cntr_ch, u_ch, u0_ch, d_ch, jm_ch, p_ch, fpc_ch, Q_ch = cmeans(all_data, ncenters, 2,
#                                  error=0.000005, maxiter=2000, var=var, method =2,init=None)
    info_gran(u, d, ncenters, dim, power)

    # Store fpc values for later
    fpcs.append(fpc)
    cluster_membership = np.argmax(u, axis=0)
#     print('cluster membership', cluster_membership)


#     print('Weighted Eculidean find cntr are', cntr)
#     print('Chebyshev find cntr are', cntr_ch)
#     base_coverage = find_base_coverage(power)* dim
#     print(base_coverage)
#     for i in range(ncenters):
# #         ax = fig.subplot(1,ncenters,i+1)
#         ax[0][i].bar(range(num_data), u_ch[i], color="blue")
#         ax[0][i].set_title('Membership value')
#     membership = []


    radius_lst, coverage_result = generate_new_coverage(u, d, power, ncenters, dim)
    normalize_dist = np.zeros((1, d.shape[1]))

    actual_d = distance_euclidean(all_data.T, cntr, var)
    for i in range(d.shape[0]):
        normalize_dist = np.vstack((normalize_dist, normalization(d[i])))
    normalize_dist = np.delete(normalize_dist, 0, 0)

    similarity(normalize_dist, radius_lst, ncenters, dim)
    original_mem = check_membership(normalize_dist, radius_lst, ncenters)
    print('radius_lst is ', radius_lst)
#     V_value = radius_lst * u[1].size

#     V_value = np.multiply(V_value, (1-radius_lst)**power)
#     print('opt radius_lst is ', V_value)
#     radius_lst_ch = np.delete(radius_lst_ch, 0)
#     print('radius_lst is', radius_lst)
#     sp = (1 - radius_lst) ** power
#     sp_ch = (1 - radius_lst_ch) ** power
#     print('sp is', sp)
#     coverage_result = np.multiply(radius_lst, sp)
#     coverage_result_ch = np.multiply(radius_lst_ch, sp_ch)
    print('coverage result', coverage_result)
    width = 0.35
    x_axis = np.array(range(ncenters))+1
    ax[1].bar(x_axis, np.multiply(coverage_result, np.power(1-radius_lst,power)), width, color="blue",  label='Weighted Eculidean')
#     ax[0].bar(x_axis+width, coverage_result_ch, width, color="r",  label='Chebyshev')
    ax[1].set_ylim([0.0, 0.5])
    ax[1].set_title('cov * sp beta={:10.2f}'.format(power))
    ax[1].legend()

    ax[0].plot(range(len(Q)), Q, color="blue", label='Weighted Eculidean')
    ax[0].set_title('Objective Function Value with m={0}'.format(fuzzy_coe))
    ax[0].legend()

#     ax[2].plot(range(len(Q_ch)), Q_ch, color="r", label='Chebyshev')
#     ax[2].set_title('Q value')
#     ax[2].legend()
#     ax[3].plot(range(len(cluster_membership)), u[1])
    ax[3].axis('off')
#     for i in range(3, ncenters):
#         ax[1][i].axis('off')

    power, power_lst, intersect_num_lst = fin_best_beta(ncenters, cntr, u, d, power, radius_lst,dim, var)
    # Plot assigned clusters, for each data point in training set
#     opti_coverage = find_base_coverage(power)* dim

    radius_lst, coverage_result = generate_new_coverage(u, d, power, ncenters, dim)
    similarity(normalize_dist, radius_lst, ncenters, dim)
    mem = check_membership(normalize_dist, radius_lst, ncenters)
    print('radius_lst is ', radius_lst)
#     V_value = radius_lst * u[1].size
#     V_value = np.multiply(V_value, (1-radius_lst)**power)
#     print('opt radius_lst is ', V_value)
#     sp = (1 - radius_lst) ** power
#     coverage_result = np.multiply(radius_lst, sp)
# #     print('the cov * sp is ', coverage_result)
    ax[2].bar(x_axis, coverage_result, width, color="blue",  label='Weighted Eculidean')
    ax[2].set_ylim([0.0, 0.5])
    ax[2].set_title('optimal cov * sp beta={:10.2f}'.format(power))
    ax[2].legend()
#     ax[2].axis('off')


    fig.tight_layout()

#     print(new_membership)
    return original_mem, mem


# In[75]:


def real_data():
    colors = ['y', 'g', 'b', 'orange']
    trainsize = 506
    testsize = 0

#     trainset, testset = dtl.load_breast_cancer(trainsize,testsize)
#     trainset, testset = dtl.echo(trainsize,testsize)
    trainset, testset = dtl.boston_house(trainsize,testsize)
#     trainset, testset = dtl.heart_hungarian(trainsize,testsize)
#     trainset, testset  = dtl.load_WDBC(trainsize, testsize)
#     trainset = np.arrary(trainset)
    Xtrain = trainset[0]
    ytrain = trainset[1]
    dim = Xtrain.shape[0]
#     print('trainset',trainset)
#     print(testset)
    if(np.isnan(np.min(Xtrain))):
#         print('trainset',Xtrain)
#         print('have nan')
#         row_mean = np.nanmean(Xtrain, axis=1)
        row_mean = []
        sum_non_nan = 0
        count = 0
        for i in range(Xtrain.shape[0]):
            sum_non_nan = 0
            count = 0
            for j in range(trainsize):
                if(not np.isnan(Xtrain[i, j])):
#                     print('no', Xtrain[i, j])
                    sum_non_nan = sum_non_nan + Xtrain[i,j]
                    count +=1
#             print(i,sum_non_nan, count)
            if(count >0):
                row_mean.append(sum_non_nan/count)
#                 row_mean.append(0)
            else:
                row_mean.append(0)

        inds = []
        for i in range(Xtrain.shape[0]):
            for j in range(trainsize):
                if(np.isnan(Xtrain[i, j])):
                    Xtrain.itemset((i,j), row_mean[i])

#         print('mean', row_mean)
#         print(inds)
#     print('xtrain',Xtrain)
    return Xtrain, colors, ytrain


# In[76]:


# xtrain, colors = data_generator(6, 2, 200)
xtrain, colors, ytrain = real_data()
original_mem, new_mem = cluster_plot_normal_data(xtrain, colors, 1, 6, xtrain.shape[0], 2)
for i in range(6):
#     print(new_mem)
    print('cluster ', i, np.mean(ytrain[original_mem[i]]), 'in best beta ', np.mean(ytrain[new_mem[i]]))
