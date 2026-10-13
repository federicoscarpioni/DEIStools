"""
06 -- A first BLTVA estimation.

One estimation of the transistor-gated cell with the settings example 14
arrives at. BLTVA does not return an impedance at a list of instants: it
returns the coefficients of a model, together with an uncertainty on them.

Run:  python examples/06_bltva_first_run.py [--save] [--no-widgets]
"""
import numpy as np
import matplotlib.pyplot as plt

from deistools.processing.bltva import (
    bltva, legendre_basis, select_na, check_identifiability,
    visualize_bltva_fit)
from deistools.simulation.models.r_rc_transistor import simulate_r_rc_transistor

from _plotting import parse_args, finish, show_all, overview, time_colours

args = parse_args(__doc__)


# --------------------------------------------------------------------------
# The measurement to analyse (see example 01)
# --------------------------------------------------------------------------
Tp = 20.0
fs = 1000.0
P = 20

harmonics = np.array([1, 3, 5, 9, 13, 17, 27, 39, 59, 89, 133, 201, 301])
frequencies = harmonics / Tp
amplitudes = np.full(frequencies.size, 0.01)
phases = np.random.default_rng(3).uniform(0, 2 * np.pi, frequencies.size)

N = int(fs * Tp) * P
T = N / fs

sim = simulate_r_rc_transistor(
    T, 1.0 / fs, frequencies, amplitudes, phases,
    R_s=10.0, R_p=30.0, C=1e-3,
    V_gate_start=1.645, V_gate_rate=(2.455 - 1.645) / T, R_T_max=70.0,
    noise_std_V=1e-4, seed=1)

time = sim["time_s"]
voltage = sim["voltage_V"]
current = sim["current_A"]
Z_true = sim["Z_true"]


# --------------------------------------------------------------------------
# The estimation
# --------------------------------------------------------------------------
# BLTVA fits Z(f,t) = sum_p H_p(f) * b_p(t), with b_p the Legendre
# polynomials, directly in the frequency domain. It uses the skirts -- the
# spread that time variation puts around each excited line -- as its data.
#
# It takes HARMONIC NUMBERS rather than frequencies, and the number of
# periods P, because it works in DFT bins throughout.

Np = 8              # order of the Legendre expansion of Z(f,t)
Nq = 2              # order of the trend polynomial at DC
band = 50           # DFT bins used either side of each tone

# The orders are not free: they share a budget set by P. check_identifiability
# returns the number of parameters a band must support, and raises if the
# record cannot support them.
n_parameters = check_identifiability(P, Np, Nq, select_na(P, Np))

result = bltva(
    voltage,                  # measured response
    current,                  # measured excitation
    fs,                       # sampling rate [Hz]
    P,                        # number of periods in the record
    harmonics,                # excited HARMONIC NUMBERS, not frequencies
    Np=Np,
    Nq=Nq,
    Na=None,                  # transient order: chosen automatically
    harmonics="excited",      # fit the excited bins only
    band_halfwidth=band,
)


# --------------------------------------------------------------------------
# What came back
# --------------------------------------------------------------------------
print("  bltva returns a BLTVAResult:")
print(f"    H_p            {str(result.H_p.shape):<12} complex, (n_freq, Np+1) coefficients")
print(f"    f_exc          {str(result.f_exc.shape):<12} the excited frequencies [Hz]")
print(f"    std_H_p        {str(result.std_H_p.shape):<12} uncertainty on each coefficient")
print(f"    diagnostics    {len(result.diagnostics):<12} one entry per band fitted")

print(f"\n  Na chosen automatically: {select_na(P, Np)}")
print(f"  {n_parameters} parameters per band, against a limit of {2 * P - 1}")
print(f"  worst condition number {result.worst_condition():.1e}")
print(f"  worst band residual    {result.worst_residual():.1e}")

# H_p[:, 0] is the time-averaged impedance -- the best stationary fit to a
# cell that was not stationary. The higher orders describe the change.
print(f"\n  H_p[:, 0] is the time average: "
      f"{np.abs(result.H_p[:, 0]).min():.1f} - "
      f"{np.abs(result.H_p[:, 0]).max():.1f} ohm")
print("   order p   max |H_p| / ohm   relative to H_0")
for p in range(result.H_p.shape[1]):
    magnitude = np.abs(result.H_p[:, p]).max()
    print(f"  {p:>8}   {magnitude:>15.4f}   "
          f"{magnitude / np.abs(result.H_p[:, 0]).max():>15.2e}")


# --------------------------------------------------------------------------
# Turning the coefficients into an impedance
# --------------------------------------------------------------------------
# Z(f,t) is a smooth polynomial in time, so it can be evaluated wherever you
# like. Here at 200 instants; all 400 000 samples would be the same curve.

instants = np.linspace(0, N - 1, 200).astype(int)
basis = legendre_basis(N, Np)              # (N, Np+1)
Z = result.H_p @ basis[instants].T         # (n_freq, 200)
t = time[instants]

print(f"\n  Z = H_p @ legendre_basis(N, Np).T  ->  {Z.shape}")
print("  the time resolution is set by the model order, not by a window:")
print(f"  an order-{Np} polynomial over {T:.0f} s")


# --------------------------------------------------------------------------
# The uncertainty
# --------------------------------------------------------------------------
# BLTVA estimates the noise from bins the excitation never touched, and
# propagates it to the coefficients. Neither STFT nor DMFA offers this.

print("\n   f / Hz        |H_0|     std(H_0)   relative")
for i in (0, 6, 12):
    magnitude = np.abs(result.H_p[i, 0])
    sigma = result.std_H_p[i, 0]
    print(f"  {frequencies[i]:>7.2f}   {magnitude:>10.3f}   {sigma:>10.2e}   "
          f"{sigma / magnitude:>8.1e}")


# --------------------------------------------------------------------------
# How good is it?
# --------------------------------------------------------------------------
truth = np.array([np.interp(t, time, Z_true[k].real)
                  + 1j * np.interp(t, time, Z_true[k].imag)
                  for k in range(frequencies.size)])
error = np.abs(Z - truth) / np.abs(truth)
predicted = np.median(result.std_H_p[:, 0] / np.abs(result.H_p[:, 0]))

print(f"\n  relative error against the known truth:")
print(f"    median {np.median(error):.2e}, worst {np.max(error):.2e}")
print(f"    predicted from noise alone: {predicted:.2e}")
print(f"    ratio {np.median(error) / predicted:.0f}")
print("\n  The error is far above what noise explains, and that is the")
print("  uncertainty doing its job. This circuit's Z(f,t) is a rational")
print("  function of time, not a polynomial, so the Legendre series")
print("  approximates it and leaves a residue. More averaging would not")
print("  help: the model is the limit, not the measurement.")


# --------------------------------------------------------------------------
# Figure 1: the standard view -- signals, spectra, estimate
# --------------------------------------------------------------------------
fig1 = overview(time, voltage, current, fs, frequencies, Z=Z, t_Z=t,
                Z_true=truth)
finish(fig1, "06_bltva_overview", args.save,
       f"BLTVA at Np = {Np}, Nq = {Nq}, band = {band}")


# --------------------------------------------------------------------------
# Figure 2: the coefficients, which are what BLTVA actually estimates
# --------------------------------------------------------------------------
fig2, axs = plt.subplots(1, 3, figsize=(15, 4.6))

axs[0].errorbar(frequencies, np.abs(result.H_p[:, 0]),
                yerr=result.std_H_p[:, 0], fmt="-o", ms=4, capsize=3,
                color="C0")
axs[0].set(xscale="log", yscale="log", xlabel="frequency / Hz",
           ylabel=r"$|H_0|$ / $\Omega$",
           title="the time average, with its uncertainty")
axs[0].grid(True, which="both", alpha=0.25)

for p in range(1, min(result.H_p.shape[1], 5)):
    axs[1].loglog(frequencies, np.abs(result.H_p[:, p]), "-o", ms=4,
                  label=f"$|H_{p}|$")
axs[1].set(xlabel="frequency / Hz", ylabel=r"$|H_p|$ / $\Omega$",
           title="the orders that describe the change")
axs[1].legend(fontsize=8)
axs[1].grid(True, which="both", alpha=0.25)

for p, colour in zip(range(min(Np + 1, 5)),
                     plt.get_cmap("viridis")(np.linspace(0, 0.85, 5))):
    axs[2].plot(time, basis[:, p], color=colour, label=f"$b_{p}(t)$")
axs[2].set(xlabel="time / s", ylabel="basis value",
           title="the basis they multiply")
axs[2].legend(fontsize=8)

finish(fig2, "06_bltva_coefficients", args.save,
       "BLTVA estimates coefficients, not an impedance per instant")


# --------------------------------------------------------------------------
# The interactive view
# --------------------------------------------------------------------------
# visualize_bltva_fit takes the same arguments as bltva. For each tone it
# shows the band data against the fitted model, and the coefficients that
# become H_p. A healthy fit has the low orders dominant and the high ones
# falling away; coefficients that grow with order mean the band is being
# fitted by cancellation between large terms.

if args.save or args.no_widgets:
    print("\n  visualize_bltva_fit(voltage, current, fs, P, harmonics, Np=...,")
    print("                      Nq=..., band_halfwidth=...)")
    print("  is an interactive slider figure. Run without --save to open it.")
else:
    print("\n  opening visualize_bltva_fit -- close the window to finish")
    visualize_bltva_fit(voltage, current, fs, P, harmonics,
                        Np=Np, Nq=Nq, harmonics="excited",
                        band_halfwidth=band)

show_all(args.save)
