import numpy as np
import skfuzzy as fuzz
from scipy.spatial.distance import cdist
from scipy import optimize
import matplotlib.pyplot as plt



def data_generator(num_cntr, data_points):
    #creating testing data

    #define the color of each cluster
    colors = ['y', 'g', 'b', 'orange']

    # Define three cluster centers

    centers = np.ndarray(shape=(num_cntr, 2), dtype=float)
    sigmas = np.ndarray(shape=(num_cntr, 2), dtype=float)
    for i in range(num_cntr):
        centers.itemset((i,0), np.random.uniform(0, 10))
        centers.itemset((i,1), np.random.uniform(0, 10))
        sigmas.itemset((i,0), np.random.uniform(0, 1.5))
        sigmas.itemset((i,1), np.random.uniform(0, 1.5))
    print('generating centers', centers)
    print('generating sigmas', sigmas)

    # centers = [[1, 1],
    #            [8, 8],
    #            [4, 4],
    #            [2,7]]
    #
    # # Define three cluster sigmas in x and y, respectively
    # sigmas = [[0.8, 0.3],
    #           [0.3, 0.5],
    #           [1.1, 0.7],
    #           [0.2, 0.5]]


    # Generate test data
    # np.random.seed(data_points)
    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.normal(xmu, xsigma, data_points)))
        ypts = np.hstack((ypts, np.random.normal(ymu, ysigma, data_points)))
        labels = np.hstack((labels, np.ones(data_points) * i))

    xpts = np.delete(xpts, 0)
    ypts = np.delete(ypts, 0)
    labels = np.delete(labels, 0)

    # Visualize the test data
    # fig0, ax0 = plt.subplots()
    # for label in range(4):
    #     ax0.plot(xpts[labels == label], ypts[labels == label], '.',
    #              color=colors[label])
    # t1= data_points*num_cntr
    # print('new data is generated')
    # ax0.set_title('data points = {1:.0f};  centers = {0}'.format(num_cntr, t1))
    # plt.show()


    return xpts, ypts, labels, colors

# In[4]:


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
#     print('center is ', cntr)
    if(method == 1):
        d = _distance(data, cntr, var)
    elif(method == 2):
        d = _distance_Chebyshev(data, cntr, var)

    # d = np.fmax(d, np.finfo(np.float64).eps)
#     print('distance is ', d)
#     print('d after transpose is ', type(data), data.shape)
#     print('d i and 2 is ', data[1], data[2])

    jm = (um * d ** 2).sum() #objective function records

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def _distance(data, centers, var):
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
        im_data = np.power(im_data, 2)/var
#         print('var is ', var)
        im_data = np.sum(im_data, axis=1)
#         print('final data is ', im_data)
        # im_data = np.sqrt(im_data)
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

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m, var, method)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc

def find_base_coverage(p):
    def f(x, p):
        return (-1)*((1-x)**p)*x
    if (p==0):
        return 1
    minimum = optimize.fmin(f, 0, args=(p,))
    print("find the min", minimum)
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

def cntr_dist_cal(ncenters, cntr, var ):
    cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    for i in range(ncenters):
        for j in range(ncenters):
            cntr_dist.itemset((i,j), np.linalg.norm(cntr[i]-cntr[j]))

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
    print('the new method dist is ', weight_euclidean)
    return weight_euclidean

# def cntr_dist_cal(ncenters, cntr, var ):
#     cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
#     for i in range(ncenters):
#         for j in range(ncenters):
# #             sub_dist = np.zeros(cntr.shape[1])
# #             print("cntr dist")
# #             print(cntr[i],cntr[j])
#             sub_dist = np.power(cntr[i]-cntr[j], 2)/var
# #             print(sub_dist)
# #             print(np.sum(sub_dist))
#             cntr_dist.itemset((i,j), np.sum(sub_dist))

    return cntr_dist

def if_intersect(cntr_dist, radius_lst, ncenters, cntr):
    intersect_num = 0
    radius_sum = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    # print(ncenters, radius_lst)
    for i in range(ncenters):
        for j in range(ncenters):
            print(i,j)
            radius_sum.itemset((i,j), (radius_lst[i] + radius_lst[j]))
    cntr_radius_diff = cntr_dist - radius_sum
    for i in range(ncenters):
        for j in range(ncenters):
            if(i != j and cntr_radius_diff[i][j]<0):
                print('intersecting between ', cntr[i], ' and ', cntr[j])
                intersect_num += 1
    if(intersect_num > 0):
        return True, intersect_num/2
    else:
        return False, intersect_num/2


def generate_new_coverage(u, d, power, ncenters):
    radius_lst = np.zeros(1)
    for j in range(ncenters):
        coverage = np.zeros(1)
#         coverage1 = np.zeros(1)
        normal_data = normalization(d[j])
        base_coverage = find_base_coverage(power)**2

#         radius_1 = np.zeros(1)
        for i in range(d[j].size):
            if(normal_data[i] < base_coverage):
                coverage = np.hstack((coverage, u[j][i]))
#                 coverage1 = np.hstack((coverage1, normal_data[i]))
        coverage = np.delete(coverage, 0)
        #             coverage1 = np.delete(coverage1, 0)
        radius_1 = np.sum(coverage) / u[j].size
        radius_lst = np.hstack((radius_lst, radius_1))
    radius_lst = np.delete(radius_lst, 0)
    return radius_lst


def fin_best_beta(ncenters, cntr, u, d, power, radius_lst, step_size, var):
    cntr_dist = np.ndarray(shape=(ncenters, ncenters), dtype=float)
    # step_size = 1
#     np.linalg.norm(sig_data - cntr_data)
    cntr_dist = cntr_dist_cal(ncenters, cntr, var)
    intersect_num_lst = []
    power_lst = []
#     print(if_intersect(cntr_dist, radius_lst, ncenters, cntr))
    # print('cntr at ',cntr)
    # print('cntr dist at ',cntr_dist)
    # print(radius_lst)
    power_lst.append(power)
    check_intersect, intersect_num = if_intersect(cntr_dist, radius_lst, ncenters, cntr)
    intersect_num_lst.append(intersect_num)
    if(check_intersect):
        while(True):
            power = power + step_size
            radius_lst, _ = generate_new_coverage_new(u, d, power, ncenters,2)
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
            radius_lst, _ = generate_new_coverage_new(u, d, power, ncenters,2)
            check_intersect, intersect_num = if_intersect(cntr_dist, radius_lst, ncenters, cntr)
            power_lst.append(power)
            intersect_num_lst.append(intersect_num)
            if(check_intersect):
                power = power + step_size
                break

    print('optimal power value is ', power)

    return power, power_lst, intersect_num_lst

def get_rectangle_cor(ncenters, cntr, radius_lst):

    rec_cor = np.zeros(shape=(ncenters, 3), dtype=float)
    for i in range(ncenters):
        rec_cor.itemset((i,0), cntr[i][0] - radius_lst[i]/2)
        rec_cor.itemset((i,1), cntr[i][1] + radius_lst[i]/2)
        rec_cor.itemset((i,2), radius_lst[i])


    return rec_cor

def if_intersect_rec(rec_cor, ncenters, cntr):
    intersect_num = 0
    for i in range(ncenters):
        for j in range(ncenters):
            if(i != j):
                if((rec_cor[i][0] + rec_cor[i][2] < rec_cor[j][0]) or
                   (rec_cor[j][0] + rec_cor[j][2] < rec_cor[i][0]) or
                   (rec_cor[i][1] - rec_cor[i][2] > rec_cor[j][1]) or
                   (rec_cor[j][1] - rec_cor[j][2] > rec_cor[i][1])):
                    pass
                else:
                    intersect_num += 1
                    print('intersecting between ', cntr[i], ' and ', cntr[j])
    if(intersect_num > 0):
        return True, intersect_num/2
    else:
        return False, intersect_num/2

def find_best_beta_chebyshev(ncenters, cntr, u, d, power, radius_lst,  step_size):
    rec_cor = get_rectangle_cor(ncenters, cntr, radius_lst)
    # step_size = 1
    print(cntr)
    print('coverage is', radius_lst)
    print(rec_cor)
    print(if_intersect_rec(rec_cor, ncenters, cntr))
    intersect_num_lst = []
    power_lst = []
    check_intersect, interct_num = if_intersect_rec(rec_cor, ncenters, cntr)
    power_lst.append(power)
    intersect_num_lst.append(interct_num)
    if(check_intersect):
        while(True):
            power = power + step_size
            radius_lst = generate_new_coverage(u, d, power, ncenters)
            rec_cor = get_rectangle_cor(ncenters, cntr, radius_lst)
            check_intersect, interct_num = if_intersect_rec(rec_cor, ncenters, cntr)
            power_lst.append(power)
            intersect_num_lst.append(interct_num)
            if(not check_intersect):
                break
    elif(not check_intersect):
        while(True):
            power = power - step_size
            if(power<0):
                power = power + step_size
                break
            radius_lst = generate_new_coverage(u, d, power, ncenters)
            rec_cor = get_rectangle_cor(ncenters, cntr, radius_lst)
            check_intersect, interct_num = if_intersect_rec(rec_cor, ncenters, cntr)
            power_lst.append(power)
            intersect_num_lst.append(interct_num)
            if(check_intersect):
                power = power + step_size
                break

    print('optimal power value is ', power)

    return power, power_lst, intersect_num_lst

def generate_new_coverage_new(u, d, power, ncenters, dim):
    radius_lst = np.zeros(1)
    cov_sp_lst = np.zeros(1)
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
        cov_sp_lst = np.hstack((cov_sp_lst, max_v/num))

    radius_lst = np.delete(radius_lst, 0)
    cov_sp_lst = np.delete(cov_sp_lst, 0)
    return radius_lst, cov_sp_lst

def info_gran(u, dist, ncenters, dim, power):
    print('info granular')
    print(dist.shape, type(dist))
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
                max_v = cov*(1-ro)
                ro_opt = ro
#                 ro += dr
#             else:
#                 break
#         radius_lst = np.hstack((radius_lst, ro_opt))
            ro += dr
        print('clster ', i, ro_opt, max_v)
