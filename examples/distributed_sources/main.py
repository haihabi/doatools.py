import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import matplotlib.pyplot as plt
from examples.distributed_sources.ds_helpers import sweep_snrs, plot_array_sources

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
#################################
#
#################################


# Create a 12-element ULA.
ula = model.UniformLinearArray(n_sensors, d0)

# We use root-MUSIC.
estimator = estimation.RootMUSIC1D(wavelength)
width = 0.1
print(width * 180)
sources = model.FarField1DGroupSourcePlacement(
    rho,
    in_total_sources=total_sources,
    in_width=[width * np.pi],
    width_estimation=width_estimation,
)

snrs = np.linspace(-20, 40, 20)

source_signal = model.ComplexStochasticSignal(sources.size, power_source)

_, mle_psi, pmle_psi, pmle_delta, _, crbs_b1, crbs_b2, crbs_b1_width, crbs_b2_width = sweep_snrs(ula, sources,
                                                                                                 source_signal,
                                                                                                 power_source, snrs,
                                                                                                 n_snapshots,
                                                                                                 wavelength, d0,
                                                                                                 estimator,
                                                                                                 in_width_estimation=width_estimation)

plt.figure(figsize=(8, 6))
plt.semilogy(
    snrs, crbs_b1, '--',
    snrs, crbs_b2,
    snrs, pmle_psi, "--x",
    snrs, mle_psi, "--o"
)
plt.xlabel('SNR [dB]', fontsize=fontsize)
plt.ylabel(r'RMSE [deg]', fontsize=fontsize)
plt.grid(True)
plt.legend(['B1', 'B2', 'MUSIC'], fontsize=fontsize)
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
raise NotImplemented
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
    pmle_psi, pmle_delta, crbs_b1, crbs_b2, crbs_b1_width, crbs_b2_width = sweep_snrs(ula, sources, source_signal,
                                                                                      power_source, snrs,
                                                                                      n_snapshots, wavelength, d0,
                                                                                      estimator,
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
