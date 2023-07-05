import numpy as np
import doatools.plotting as doaplot
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import matplotlib.pyplot as plt


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


def sweep_snrs(in_array, in_sources, in_source_signal, in_snrs, in_n_snapshots, in_n_repeats=1000, plot_music=False,
               in_width_estimation=False):
    mses = np.zeros((len(in_snrs),))
    _mses_rho = np.zeros((len(in_snrs),))
    _crbs_b1 = np.zeros((len(in_snrs),))
    _crbs_b2 = np.zeros((len(in_snrs),))
    _crbs_b1_width = np.zeros((len(in_snrs),))
    _crbs_b2_width = np.zeros((len(in_snrs),))
    S = in_source_signal.emit(in_n_snapshots)
    A = in_array.steering_matrix(in_sources, wavelength)
    for i, snr in enumerate(in_snrs):
        power_noise = power_source / (10 ** (snr / 10))
        noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
        # The squared errors and the deterministic CRB varies
        # for each run. We need to compute the average.
        cur_mse = 0.0
        cur_mse_rho = 0.0

        for r in range(in_n_repeats):
            # Stochastic signal model.
            N = noise_signal.emit(in_n_snapshots)
            Y = A @ S + N
            Ry = (Y @ Y.conj().T) / in_n_snapshots
            resolved, estimates = estimator.estimate(Ry, in_sources.size, d0)
            if r == 0 and i == 0 and plot_music:
                # Get the estimated covariance matrix.
                _, R = model.get_narrowband_snapshots(ula, in_sources, wavelength, source_signal, noise_signal,
                                                      in_n_snapshots, return_covariance=True)
                # Create a MUSIC-based estimator.
                grid = estimation.FarField1DSearchGrid()
                estimator_music = estimation.MUSIC(ula, wavelength, grid)

                # Get the estimates.
                _, _, sp = estimator_music.estimate(R, in_sources.size, return_spectrum=True)

                # Plot the MUSIC-spectrum.
                doaplot.plot_spectrum({'MUSIC': sp}, grid, ground_truth=in_sources, use_log_scale=True)
                plt.savefig(f"music_{snr}.svg")

            # In practice, you should check if `resolved` is true.
            # We skip the check here.
            cur_mse += np.mean((estimates.locations - in_sources.locations) ** 2)
            if in_width_estimation:
                cur_mse_rho += np.mean((in_sources.h_dagger @ estimates.locations - in_sources.rho) ** 2)
            else:
                cur_mse_rho += np.mean((in_sources.h_dagger @ estimates.locations - in_sources.rho) ** 2)

        B_det, FIM = perf.crb_sto_farfield_1d(ula, in_sources, wavelength, 1,
                                              power_noise, n_snapshots)

        B1 = np.linalg.inv(in_sources.h_matrix.T @ FIM @ in_sources.h_matrix)
        B2 = in_sources.h_dagger @ B_det @ in_sources.h_dagger.T
        mses[i] = cur_mse / in_n_repeats
        _mses_rho[i] = cur_mse_rho / in_n_repeats
        if in_width_estimation:
            b1_diag = np.diag(B1)
            b2_diag = np.diag(B2)
            _crbs_b1[i] = bound2degree(np.mean(b1_diag[:n_groups]))
            _crbs_b2[i] = bound2degree(np.mean(b2_diag[:n_groups]))
            _crbs_b1_width[i] = bound2degree(np.mean(b1_diag[n_groups:]))
            _crbs_b2_width[i] = bound2degree(np.mean(b2_diag[n_groups:]))
        else:
            _crbs_b1[i] = np.mean(np.diag(B1))
            _crbs_b2[i] = np.mean(np.diag(B2))
        print('Completed SNR = {0:.2f} dB'.format(snr))
    return _mses_rho, _crbs_b1, _crbs_b2, _crbs_b1_width, _crbs_b2_width


#################################
#
#################################
fontsize = 16
n_snapshots = 300
n_sensors = 12
wavelength = 1.0  # normalized
d0 = wavelength / 2
power_source = 1  # Normalized
edge_position = np.pi / 4
total_sources = 4
n_groups = 1
width_estimation = True
rho = np.linspace(-edge_position, edge_position, n_groups)
print(edge_position / np.pi * 180)
# We use root-MUSIC.
estimator = estimation.RootMUSIC1D(wavelength)
# Create a 12-element ULA.
ula = model.UniformLinearArray(n_sensors, d0)
width = 0.1
print(width * 180)
sources = model.FarField1DGroupSourcePlacement(
    rho,
    in_total_sources=total_sources,
    in_width=[width * np.pi],
    width_estimation=width_estimation,
)

snrs = np.linspace(-20, 40, 30)
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
_, crbs_b1, crbs_b2, crbs_b1_width, crbs_b2_width = sweep_snrs(ula, sources, source_signal, snrs, n_snapshots,
                                                               in_width_estimation=width_estimation)
plt.figure(figsize=(8, 6))
plt.semilogy(
    snrs, crbs_b1, '--',
    snrs, crbs_b2,
)
plt.xlabel('SNR [dB]', fontsize=fontsize)
plt.ylabel(r'RMSE [deg]', fontsize=fontsize)
plt.grid(True)
plt.legend(['B1', 'B2'], fontsize=fontsize)
# plt.title('MSE vs. CRB')
plt.tight_layout()
plt.savefig("mse_vs_snr_doa.svg")
plt.show()
if width_estimation:
    plt.figure(figsize=(8, 6))
    plt.semilogy(
        snrs, crbs_b1_width, '--',
        snrs, crbs_b2_width,
    )
    plt.xlabel('SNR [dB]', fontsize=fontsize)
    plt.ylabel(r'RMSE [deg]', fontsize=fontsize)
    plt.grid(True)
    plt.legend(['B1', 'B2'], fontsize=fontsize)
    # plt.title('MSE vs. CRB')
    plt.tight_layout()
    plt.savefig("mse_vs_snr_width.svg")
    plt.show()

snrs = [20]

res = []
k = 30
width_array = np.linspace(0.005, 0.15, k)
for j, width in enumerate(width_array):
    sources = model.FarField1DGroupSourcePlacement(
        rho,
        in_total_sources=total_sources,
        in_width=[width * np.pi],
        width_estimation=width_estimation,

    )
    if j == k // 2:
        plot_array_sources(ula, sources, rho, file_name="location.svg")
    # All sources share the same power.
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    _, crbs_b1, crbs_b2, crbs_b1_width, crbs_b2_width = sweep_snrs(ula, sources, source_signal, snrs, n_snapshots,
                                                                   in_width_estimation=width_estimation)
    # print(crbs_b1, crbs_b2, width)
    res.append([crbs_b1, crbs_b2, crbs_b1_width, crbs_b2_width])
res = np.asarray(res)
plt.figure(figsize=(8, 6))
nan_index = np.where(np.isnan(res[:, 1, 0]))[0][-1] + 1
print(nan_index)
plt.semilogy(
    180 * width_array, res[:, 0, 0], '--',
    180 * width_array[nan_index:], res[:, 1, 0][nan_index:],
)
plt.plot([180 * width_array[nan_index], 180 * width_array[nan_index]],
         [np.min(res[:, 0, 0]), np.max(res[:, 1, 0][nan_index:])])
plt.xlabel(r'$\delta$ [deg]', fontsize=fontsize)
plt.ylabel(r'RMSE [deg]', fontsize=fontsize)
plt.grid(True)
plt.legend(['B1', 'B2'], fontsize=fontsize)
# plt.title('MSE vs. CRB')
plt.tight_layout()
plt.savefig("mse_vs_width_doa.svg")
plt.show()
if width_estimation:
    plt.figure(figsize=(8, 6))
    # nan_index = np.where(np.isnan(res[:, 3, 0]))[0][-1]
    plt.semilogy(
        180 * width_array, res[:, 2, 0], '--',
        180 * width_array[nan_index:], res[:, 3, 0][nan_index:],
    )
    plt.plot([180 * width_array[nan_index], 180 * width_array[nan_index]],
             [np.min(res[:, 2, 0]), np.max(res[:, 3, 0][nan_index:])])

    plt.xlabel(r'$\delta$  [deg]', fontsize=fontsize)
    plt.ylabel(r'RMSE  [deg]', fontsize=fontsize)
    plt.grid(True)
    plt.legend(['B1', 'B2'], fontsize=fontsize)
    # plt.title('MSE vs. CRB')
    plt.tight_layout()
    plt.savefig("mse_vs_width_width.svg")
    plt.show()
