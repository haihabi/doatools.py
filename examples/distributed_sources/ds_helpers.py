import numpy as np
import doatools.plotting as doaplot
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import matplotlib.pyplot as plt
from tqdm import tqdm

N_GROUPS = 1


def plot_array_sources(in_array, in_sources, in_rho, file_name=None, far_field_point=20):
    plt.figure()
    loc = np.asarray([[far_field_point * np.cos(doa), far_field_point * np.sin(doa)] for doa in in_sources._locations])

    m = int(loc.shape[0] / len(in_rho))
    color_array = ["red", "green"]
    for i, _rho in enumerate(in_rho):
        plt.plot([0, far_field_point * np.sin(_rho)], [0, far_field_point * np.cos(_rho)], "--", label=f"Group {i} DoA",
                 color=color_array[i])
        plt.plot(loc[m * i:(i + 1) * m, 1], loc[m * i:(i + 1) * m, 0], 'x', label=f"Group {i} Sources",
                 color=color_array[i])

    plt.scatter(in_array._locations[:, 0], np.zeros((in_array._locations.shape[0])), label="Array", color="blue")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()


def bound2degree(in_b):
    return 180 * np.sqrt(in_b) / np.pi


def sweep_snrs(in_array,
               in_sources,
               in_source_signal,
               in_power_source,
               in_snrs,
               in_n_snapshots,
               in_wavelength,
               in_d0,
               in_estimator,
               in_n_repeats=1000,
               plot_music=False,
               in_width_estimation=False):
    mses = np.zeros((len(in_snrs),))
    _mses_psi_proj = np.zeros((len(in_snrs),))
    _mses_psi = np.zeros((len(in_snrs),))
    _mses_delta = np.zeros((len(in_snrs),))
    _crbs = np.zeros((len(in_snrs),))
    _crbs_b1 = np.zeros((len(in_snrs),))
    _crbs_b2 = np.zeros((len(in_snrs),))
    _crbs_b1_width = np.zeros((len(in_snrs),))
    _crbs_b2_width = np.zeros((len(in_snrs),))

    for i, snr in enumerate(in_snrs):
        power_noise = in_power_source / (10 ** (snr / 10))
        noise_signal = model.ComplexStochasticSignal(in_array.size, power_noise)
        # The squared errors and the deterministic CRB varies
        # for each run. We need to compute the average.
        cur_mse = 0.0
        cur_mse_psi_proj = 0.0
        cur_mse_psi = 0.0
        cur_mse_delta = 0.0

        for r in tqdm(range(in_n_repeats)):
            # Stochastic signal model.
            A = in_array.steering_matrix(in_sources, in_wavelength)
            S = in_source_signal.emit(in_n_snapshots)
            N = noise_signal.emit(in_n_snapshots)
            Y = A @ S + N
            Ry = (Y @ Y.conj().T) / in_n_snapshots
            resolved, estimates = in_estimator.estimate(Ry, in_sources.size, in_d0)
            resolved, estimates_one = in_estimator.estimate(Ry, N_GROUPS, in_d0)
            if r == 0 and i == 0 and plot_music:
                # Get the estimated covariance matrix.
                _, R = model.get_narrowband_snapshots(in_array, in_sources, in_wavelength, in_source_signal,
                                                      noise_signal,
                                                      in_n_snapshots, return_covariance=True)
                # Create a MUSIC-based estimator.
                grid = estimation.FarField1DSearchGrid()
                estimator_music = estimation.MUSIC(in_array, in_wavelength, grid)

                # Get the estimates.
                _, _, sp = estimator_music.estimate(R, in_sources.size, return_spectrum=True)

                # Plot the MUSIC-spectrum.
                doaplot.plot_spectrum({'MUSIC': sp}, grid, ground_truth=in_sources, use_log_scale=True)
                plt.savefig(f"music_{snr}.svg")

            # In practice, you should check if `resolved` is true.
            # We skip the check here.
            cur_mse += np.mean((estimates.locations - in_sources.locations) ** 2)
            if in_width_estimation:
                psi_delta = in_sources.h_dagger @ estimates.locations
                psi = psi_delta[0]
                delta = psi_delta[1]
                cur_mse_psi_proj += np.mean((psi - in_sources.rho[0]) ** 2)
                cur_mse_delta += np.mean((delta - in_sources.rho[1]) ** 2)

                cur_mse_psi += np.mean((estimates_one.locations - in_sources.rho[0]) ** 2)
            else:
                cur_mse_psi_proj += np.mean((in_sources.h_dagger @ estimates.locations - in_sources.rho) ** 2)

        B_det, FIM = perf.crb_sto_farfield_1d(in_array, in_sources, in_wavelength, 1,
                                              power_noise, in_n_snapshots)

        B1 = np.linalg.inv(in_sources.h_matrix.T @ FIM @ in_sources.h_matrix)
        B2 = in_sources.h_dagger @ B_det @ in_sources.h_dagger.T
        mses[i] = cur_mse / in_n_repeats
        _crbs[i] = np.mean(np.diag(B_det))
        _mses_psi_proj[i] = bound2degree(cur_mse_psi_proj / in_n_repeats)
        _mses_psi[i] = bound2degree(cur_mse_psi / in_n_repeats)
        _mses_delta[i] = bound2degree(cur_mse_delta / in_n_repeats)
        if in_width_estimation:
            b1_diag = np.diag(B1)
            b2_diag = np.diag(B2)
            _crbs_b1[i] = bound2degree(np.mean(b1_diag[:N_GROUPS]))
            _crbs_b2[i] = bound2degree(np.mean(b2_diag[:N_GROUPS]))
            _crbs_b1_width[i] = bound2degree(np.mean(b1_diag[N_GROUPS:]))
            _crbs_b2_width[i] = bound2degree(np.mean(b2_diag[N_GROUPS:]))
        else:
            _crbs_b1[i] = np.mean(np.diag(B1))
            _crbs_b2[i] = np.mean(np.diag(B2))
        print('Completed SNR = {0:.2f} dB'.format(snr))
    return mses,_mses_psi, _mses_psi_proj, _mses_delta, _crbs, _crbs_b1, _crbs_b2, _crbs_b1_width, _crbs_b2_width
