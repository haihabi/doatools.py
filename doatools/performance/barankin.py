import numpy as np
from doatools.model import FarField1DSourcePlacement
from doatools.performance.utils import unify_p_to_matrix


def compute_r_matrix(in_array, in_wavelength, in_locations, in_sigma, source_covariance):
    m = in_array.size
    if isinstance(in_locations, np.ndarray):
        sources = FarField1DSourcePlacement(in_locations.flatten())
    else:
        sources = in_locations
    A = in_array.steering_matrix(sources, in_wavelength, False, 'all')
    A_H = A.conj().T
    return (A @ source_covariance) @ A_H + in_sigma * np.eye(m)


# def search_test_point(sources):
#     pass

def search_test_points(array, wavelength, sources, sigma, p, r_inv, r_det, l_select, l_options=10000, eps=1e-2):
    k = sources.size
    # Generate test option
    if isinstance(sources, FarField1DSourcePlacement):
        base_array = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, np.ceil(np.power(l_options, 1 / k)).astype("int"))
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
        r_test_det = np.linalg.det(r_test)
        r_delta = 2 * np.linalg.inv(r_test) - r_inv
        r_delta_det = np.linalg.det(r_delta)
        r_det_temp.append([r_test_det, r_delta_det])
    r_det_temp = np.asarray(r_det_temp)

    cost = np.real(r_det / ((r_det_temp[:, 0] ** 2) * r_det_temp[:, 1]))

    index_select = np.argsort(cost)[:l_select]
    return test_points_search_array[:, index_select]


def compute_barakin_matrix(r_det, r_inv, in_test_points):
    pass


def barankin_stouc_farfield_1d(array, sources, wavelength, p, sigma, n_snapshots=1,
                               return_mode='full', l_test_points=10):
    r"""Computes the stochastic CRB for 1D farfield sources with the assumption
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
        return_mode (str): Can be one of the following:

            1. ``'full'``: returns the full CRB matrix.
            2. ``'diag'``: returns only the diagonals of the CRB matrix.
            3. ``'mean_diag'``: returns the mean of the diagonals of the CRB
               matrix.

            Default value is ``'full'``.

    Returns:
        Depending on ``'return_mode'``, can be the full CRB matrix, the
        diagonals of the CRB matrix, or the mean of the diagonals of the CRB
        matrix.

    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.

        [2] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao Bound,"
        IEEE Transactions on Signal Processing, vol. 65, no. 4, pp. 933-946,
        Feb. 2017.

        [3] C-L. Liu and P. P. Vaidyanathan, "Cramér-Rao bounds for coprime and
        other sparse arrays, which find more sources than sensors," Digital
        Signal Processing, vol. 61, pp. 43-61, 2017.
    """
    if not isinstance(sources, FarField1DSourcePlacement):
        raise ValueError('Sources must be far-field and 1D.')
    k = sources.size
    p = unify_p_to_matrix(p, k)
    m = array.size

    # Compute steering matrix of sources.

    R = compute_r_matrix(array, wavelength, sources, sigma, p)
    r_det = np.linalg.det(R)
    r_inv = np.linalg.inv(R)
    test_points = search_test_points(array, wavelength, sources, sigma, p, r_inv, r_det, l_test_points, l_options=10000,
                                     eps=1e-2)

    r_det_array = []
    r_inv_list = []
    for i in range(test_points.shape[-1]):
        r = compute_r_matrix(array, wavelength, test_points[:, i].reshape([-1, 1]), sigma, p)
        r_inv = np.linalg.inv(r)
        r_inv_list.append(r_inv)
        r_det_array.append(np.linalg.det(r))
    #
    # b_matrix = np.zeros(test_points.shape[-1], test_points.shape[-1])
    # for i in range(test_points.shape[-1]):
    #     for j in range(test_points.shape[-1]):
    #         if i < j:
    #             r_delta=
    #             den = r_det_array[i] * r_det_array[j]
    #             b_matrix[i, j] =

    # b_matrix = compute_barakin_matrix()

# r_det_test_point = [
#     np.linalg.det(compute_r_matrix(array, wavelength, test_points_list[:, i].reshape([-1, 1]), sigma, p)) for i in
#     range(l_options)]
# print("a")

# compute_barakin_matrix_diagonal()
# print("a")

# A = array.steering_matrix(sources, wavelength, False, 'all')
# A_H = A.conj().T
# R = (A @ p) @ A_H + sigma * np.eye(m)

# R_inv = np.linalg.inv(R)
# R_inv = 0.5 * (R_inv + R_inv.conj().T)
# DRD = DA_H @ R_inv @ DA
# DRA = DA_H @ R_inv @ A
# ARD = A_H @ R_inv @ DA
# ARA = A_H @ R_inv @ A
# PP = np.outer(p, p)
# FIM_tt = 2.0 * ((DRD.T * ARA + DRA.conj() * ARD) * PP).real
# FIM_pp = (ARA.conj().T * ARA).real
# R_inv2 = R_inv @ R_inv
# FIM_ss = np.trace(R_inv2).real
# FIM_tp = 2.0 * (DRA.conj() * (p[:, np.newaxis] * ARA)).real
# FIM_ts = 2.0 * (p * np.sum(DA.conj() * (R_inv2 @ A), axis=0)).real[:, np.newaxis]
# FIM_ps = np.sum(A.conj() * (R_inv2 @ A), axis=0).real[:, np.newaxis]
# FIM = np.block([
#     [FIM_tt, FIM_tp, FIM_ts],
#     [FIM_tp.conj().T, FIM_pp, FIM_ps],
#     [FIM_ts.conj().T, FIM_ps.conj().T, FIM_ss]
# ])
# CRB = np.linalg.inv(FIM)[:k, :k] / n_snapshots
# return reduce_output_matrix(0.5 * (CRB + CRB.T), return_mode), FIM
