import numpy as np
from doatools.model import FarField1DSourcePlacement
from doatools.performance.utils import unify_p_to_matrix
import scipy


def compute_r_matrix(in_array, in_wavelength, in_locations, in_sigma, source_covariance):
    m = in_array.size
    if isinstance(in_locations, np.ndarray):
        sources = FarField1DSourcePlacement(in_locations.flatten())
    else:
        sources = in_locations
    A = in_array.steering_matrix(sources, in_wavelength, False, 'all')
    A_H = A.conj().T
    return (A @ source_covariance) @ A_H + in_sigma * np.eye(m)


def search_test_points(array, wavelength, sources, sigma, p, n_snapshots, r_inv, r_det, l_select, l_options=10000,
                       eps=1e-2, range_min=-np.pi / 2, range_max=np.pi / 2):
    k = sources.size
    # Generate test option
    if isinstance(sources, FarField1DSourcePlacement):
        base_array = np.linspace(range_min + eps, range_max - eps, np.ceil(np.power(l_options, 1 / k)).astype("int"))
        test_points_search_array = np.meshgrid(*[base_array for _ in range(k)])
        test_points_search_array = np.stack(test_points_search_array)
        test_points_search_array = np.reshape(test_points_search_array, [k, -1])
        l_options = test_points_search_array.shape[-1]
    else:
        raise NotImplemented
    # Compute cost

    r_det_temp = []
    for i in range(l_options):
        r_test = compute_r_matrix(array, wavelength, test_points_search_array[:, i].reshape([-1, 1]), sigma, p)
        r_test_det = np.real(np.linalg.det(r_test))
        r_delta = 2 * np.linalg.inv(r_test) - r_inv
        r_delta_det = np.real(np.linalg.det(r_delta))
        r_det_temp.append([r_test_det, r_delta_det])
    r_det_temp = np.asarray(r_det_temp)

    cost = np.real(r_det / ((r_det_temp[:, 0] ** 2) * r_det_temp[:, 1]))

    # print("a")
    if np.any(cost < 0):
        test_points_search_array = test_points_search_array[:, cost >= 0]
        cost = cost[cost >= 0]

    peaks = scipy.signal.find_peaks(1 / cost)[0]
    return test_points_search_array[:, peaks]


def barankin_stouc_farfield_1d(array, sources, wavelength, p, sigma, n_snapshots=1, l_test_points=30):
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

    R = compute_r_matrix(array, wavelength, sources, sigma, p)
    r_det = np.real(np.linalg.det(R))
    r_inv = np.linalg.inv(R)
    test_points = search_test_points(array, wavelength, sources, sigma, p, n_snapshots, r_inv, r_det, l_test_points,
                                     l_options=1600,
                                     eps=1e-2)
    #############################
    # Compute the Barankin matrix
    #############################
    r_det_array = []
    r_inv_list = []
    for i in range(test_points.shape[-1]):
        tp = test_points[:, i].reshape([-1, 1])
        _r = compute_r_matrix(array, wavelength, tp, sigma, p)
        r_inv_list.append(np.linalg.inv(_r))
        r_det_array.append(np.real(np.linalg.det(_r)))

    b_matrix = np.zeros([test_points.shape[-1], test_points.shape[-1]])
    for i in range(test_points.shape[-1]):
        for j in range(test_points.shape[-1]):
            if i <= j:
                r_delta = np.real(np.linalg.det(r_inv_list[i] + r_inv_list[j] - r_inv))
                b_matrix[i, j] = b_matrix[j, i] = np.power(r_det / (r_delta * r_det_array[j] * r_det_array[i]),
                                                           n_snapshots)
    b_matrix = np.real(b_matrix)
    one_one = np.ones_like(b_matrix)

    delta_tp = test_points - sources.locations.reshape([-1, 1])
    delta_matrix = b_matrix - one_one
    index = np.sum(delta_matrix, axis=1) != 0
    if not np.all(index):
        print("Filter")
        delta_matrix = delta_matrix[index, index].reshape([1, 1])
        delta_tp = delta_tp[:, index]
    bound = delta_tp @ np.linalg.inv(delta_matrix) @ delta_tp.T
    return bound, b_matrix, test_points
