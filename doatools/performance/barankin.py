import matplotlib.pyplot as plt
import numpy as np

import doatools.model
from doatools.model import FarField1DSourcePlacement
from doatools.performance.utils import unify_p_to_matrix
import scipy
import doatools.model as model
from doatools import estimation
import doatools.plotting as doaplt
import doatools


def compute_r_matrix(in_array, in_wavelength, in_locations, in_sigma, source_covariance):
    m = in_array.size
    if isinstance(in_locations, np.ndarray):
        sources = FarField1DSourcePlacement(in_locations.flatten())
    else:
        sources = in_locations
    A = in_array.steering_matrix(sources, in_wavelength, False, 'all')
    A_H = A.conj().T
    # N = in_sigma / np.abs(in_sigma.max())
    return (A @ source_covariance) @ A_H + in_sigma


def search_test_points_r(in_sources, in_wavelength, in_array, max_per_dim=1, eps=1e-3):
    base_array = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, 10000)
    tp_list = []
    cost_list = []
    A = in_array.steering_matrix(in_sources, in_wavelength)
    for d in range(in_sources.size):
        res_list = []
        for _theta in base_array:
            doas = np.copy(in_sources.locations)
            doas[d] = _theta
            sources = doatools.model.FarField1DSourcePlacement(doas)

            _A = in_array.steering_matrix(sources, in_wavelength)
            v = np.abs(np.real(_A.T.conj() @ A))
            res_list.append(v)
        bp = np.asarray(res_list).flatten()
        normalized_bp = bp / np.max(bp)
        peaks = scipy.signal.find_peaks(normalized_bp)[0]

        peaks = peaks[np.flip(np.argsort(normalized_bp[peaks]))[1:3]]
        peaks = peaks[np.argmin((base_array[peaks] - in_sources.locations.flatten()) ** 2)]
        # for p in peaks:
        #     doas = np.copy(in_sources.locations)
        #     doas[d] = base_array[p]
        #     tp_list.append(doas.flatten())
        doas = np.copy(in_sources.locations)
        doas[d] = base_array[peaks]

        plt.plot(base_array.flatten() - in_sources.locations.flatten(), normalized_bp)
        plt.plot(base_array[peaks].flatten() - in_sources.locations.flatten(), normalized_bp[peaks], "v")
        plt.grid()
        plt.show()
        tp_list.append(doas.flatten())
        # tp_list.append(sources.locations.flatten())
        cost_list.append(normalized_bp)

    return np.stack(tp_list, axis=-1), cost_list, [base_array]


def search_iteration(in_range, theta_dim, current_dim, r_inv, r_det, array, wavelength, sources, sigma, p):
    test_points_search_array = np.zeros([theta_dim, in_range.shape[0]])
    test_points_search_array[current_dim, :] = in_range
    for kj in range(theta_dim):
        if kj != current_dim:
            test_points_search_array[kj, :] = sources.locations[kj]

    l_options = test_points_search_array.shape[-1]

    r_det_temp = []
    for i in range(l_options):
        r_test = compute_r_matrix(array, wavelength, test_points_search_array[:, i].reshape([-1, 1]), sigma, p)
        r_test_det = np.real(np.linalg.det(r_test))
        r_delta = 2 * np.linalg.inv(r_test) - r_inv
        r_delta_det = np.real(np.linalg.det(r_delta))
        r_det_temp.append([r_test_det, r_delta_det])
    r_det_temp = np.asarray(r_det_temp)

    cost = np.real(r_det / ((r_det_temp[:, 0] ** 2) * r_det_temp[:, 1]))

    if np.any(cost < 1):
        test_points_search_array = test_points_search_array[:, cost > 1]
        cost = cost[cost > 1]
    delta_tp = ((test_points_search_array - sources.locations.reshape([-1, 1])) ** 2).flatten()
    one_d_cost = delta_tp / (cost - 1)
    p = np.argmax(one_d_cost)
    best_cost = one_d_cost[p]

    return test_points_search_array[:, p].reshape([-1, 1]), test_points_search_array, one_d_cost, best_cost
    # print("a")


def search_test_points(array, wavelength, sources, sigma, p, n_snapshots, r_inv, r_det, l_select, l_options=10000,
                       eps=1e-2, range_min=-np.pi / 2, range_max=np.pi / 2):
    k = sources.size
    n = 1
    iter_size = int(l_options / n)
    base_array = np.linspace(range_min + eps, range_max - eps, iter_size)

    test_points_search_array_list = []
    cost_list = []
    tp_list = []
    best_cost = 0
    best_tp = 0
    for ki in range(k):
        for _ in range(n):
            tp_select, tp_search_space, cost, _best_cost = search_iteration(base_array, k, ki, r_inv, r_det, array,
                                                                            wavelength,
                                                                            sources,
                                                                            sigma, p)
            if best_cost < _best_cost:
                best_cost = _best_cost
                best_tp = tp_select
            if tp_search_space.shape[1] == iter_size:
                if (range_min + eps) == tp_select[ki, 0] or (range_max - eps) == tp_select[ki, 0]:
                    break
                else:
                    break
            else:
                delta = np.abs(np.diff(base_array)).min() / 10

                lower_index = np.where(tp_search_space[ki, 0] == base_array)[0]
                if lower_index == 0:
                    lower = base_array[0] - 2 * delta
                else:
                    lower = base_array[lower_index - 1]

                upper_index = np.where(tp_search_space[ki, -1] == base_array)[0]
                if upper_index == (iter_size - 1):
                    upper = base_array[-1] + 2 * delta
                else:
                    upper = base_array[upper_index + 1]
                base_array = np.linspace(lower + delta, upper - delta, iter_size).flatten()
                # raise NotImplemented
        tp_list.append(best_tp)
        test_points_search_array_list.append(tp_search_space)
        cost_list.append(cost)

    return np.concatenate(tp_list, axis=-1), cost_list, test_points_search_array_list


def barankin_stouc_farfield_1d(array, sources, wavelength, p, sigma, n_snapshots=1, max_test_points=2,
                               l_search_point=3200, output_search_landscape=False):
    r"""Computes the stochastic Barankin for 1D farfield sources with the assumption
    that the sources are uncorrelated.

    Under the stochastic signal model, the source signal is assumed to be
    uncorrelated complex Gaussian, and the noise signal is assumed complex
    white Gaussian. The unknown parameters include:

    * Source locations.
    * Diagonals of the source covariance matrix.
    * Noise variance.

    This function only computes the CRB for source locations.
    Because the unknown parameters do not include array perturbation parameters,
    all array perturbation parameters are assumed known during the computation.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        sources (~doatools.model.sources.FarField1DSourcePlacement):
            Source locations.
        p (float or ~numpy.ndarray): The power of the source signals. Can be

            1. A scalar if all sources are uncorrelated and share the same
               power.
            2. A 1D numpy array if all sources are uncorrelated but have
               different powers.
            3. A 2D numpy array representing the source covariance matrix.
               Only the diagonal elements will be used.

        sigma (float): Variance of the additive noise.
        n_snapshots (int): Number of snapshots. Default value is 1.


    Returns:
        Depending on ``'return_mode'``, can be the full CRB matrix, the
        diagonals of the CRB matrix, or the mean of the diagonals of the CRB
        matrix.

    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.

    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    p = unify_p_to_matrix(p, k)

    # Compute steering matrix of sources.
    if np.isscalar(sigma):
        sigma = np.eye(array.size) * sigma
    R = compute_r_matrix(array, wavelength, sources, sigma, p)
    r_det = (np.linalg.det(R))
    r_inv = np.linalg.inv(R)
    if False:
        test_points, search_landscape, test_points_search_array = search_test_points_r(sources, wavelength, array,
                                                                                       max_per_dim=max_test_points)
    else:
        test_points, search_landscape, test_points_search_array = search_test_points(array, wavelength, sources, sigma,
                                                                                     p,
                                                                                     n_snapshots, r_inv, r_det, None)

    #############################
    # Compute the Barankin matrix
    #############################
    r_det_array = []
    r_inv_list = []
    for i in range(test_points.shape[-1]):
        tp = test_points[:, i].reshape([-1, 1])
        _r = compute_r_matrix(array, wavelength, tp, sigma, p)
        r_inv_list.append(np.linalg.inv(_r))
        r_det_array.append((np.linalg.det(_r)))

    b_matrix = np.zeros([test_points.shape[-1], test_points.shape[-1]])
    for i in range(test_points.shape[-1]):
        for j in range(test_points.shape[-1]):
            if i <= j:
                r_delta = (np.linalg.det(r_inv_list[i] + r_inv_list[j] - r_inv))
                b_matrix[i, j] = b_matrix[j, i] = np.power(r_det / (r_delta * r_det_array[j] * r_det_array[i]),
                                                           n_snapshots)

    b_matrix = np.real(b_matrix)
    status = (b_matrix.diagonal() > 0)
    b_matrix = b_matrix[status.reshape([-1, 1]) @ status.reshape([1, -1])].reshape([np.sum(status), np.sum(status)])
    test_points = test_points[:, status]

    one_one = np.ones_like(b_matrix)
    delta_tp = test_points - sources.locations.reshape([-1, 1])
    delta_matrix = b_matrix - one_one
    index = np.sum(delta_matrix, axis=1) != 0
    if not np.all(index):
        print("Filter")
        delta_matrix = delta_matrix[index, index].reshape([1, 1])
        delta_tp = delta_tp[:, index]
    bound = delta_tp @ np.linalg.inv(delta_matrix) @ delta_tp.T
    if output_search_landscape:
        return bound, b_matrix, test_points, search_landscape, test_points_search_array
    else:
        return bound, b_matrix, test_points
