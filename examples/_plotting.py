"""
Figure layout helpers for the examples.

Nothing here does any science: it is matplotlib boilerplate, kept out of the
example scripts so that the simulation and the estimator calls are not buried
under sixty lines of axis formatting. Everything takes plain numpy arrays.

Two standard figures:

    overview(...)     three columns -- raw signals, spectra, impedance
    comparison(...)   two panels -- against ground truth, and relative error

Conventions: black is ground truth, time runs blue to red, Nyquist plots show
-Im(Z) against Re(Z) on equal axes.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")


def parse_args(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--save", action="store_true",
                   help="write figures to examples/figures/ instead of showing them")
    p.add_argument("--no-widgets", action="store_true",
                   help="skip the interactive slider figures")
    return p.parse_args()


def finish(fig, name, save, title=None):
    """Save the figure to examples/figures/, or leave it open to be shown."""
    if title:
        fig.suptitle(title)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    if save:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        path = os.path.join(FIGURE_DIR, f"{name}.png")
        fig.savefig(path, dpi=130)
        print(f"  figure -> {os.path.relpath(path)}")
        plt.close(fig)


def show_all(save):
    if not save:
        plt.show()


def time_colours(n):
    """n colours running blue (early) to red (late)."""
    return plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, n))


def envelope(ax, t, y, color="C0"):
    """Draw a long record as a min/max envelope, one pair per pixel bucket.

    A 400 000-sample multisine cannot be drawn sample by sample, and
    decimating it aliases the high tones into nonsense.
    """
    n = int(min(1200, y.size))
    edges = np.linspace(0, y.size, n + 1).astype(int)
    lo = np.minimum.reduceat(y, edges[:-1])
    hi = np.maximum.reduceat(y, edges[:-1])
    ax.fill_between(t[edges[:-1]], lo, hi, color=color, alpha=0.75, lw=0)


def spectrum(ax, signal, fs, frequencies, color="C0"):
    """Single-sided magnitude spectrum with the excited bins marked."""
    n = signal.size
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    bins = np.round(frequencies * n / fs).astype(int)
    ax.loglog(f[1:], mag[1:], color=color, lw=0.6, alpha=0.7)
    ax.loglog(frequencies, mag[bins], "o", ms=4, color="C3", zorder=5,
              label="excited tones")
    ax.set_xlim(frequencies.min() * 0.5, frequencies.max() * 3)
    peak = float(mag[1:].max())
    ax.set_ylim(peak * 1e-9, peak * 3)


def nyquist_vs_time(ax, Z, times, Z_true=None, n_curves=10):
    """Nyquist spectra at a series of instants, blue to red, truth in black."""
    picks = np.unique(np.linspace(0, Z.shape[1] - 1, n_curves).astype(int))
    if Z_true is not None:
        for n, j in enumerate(picks):
            ax.plot(Z_true[:, j].real, -Z_true[:, j].imag, "-", color="black",
                    lw=1.2, alpha=0.55, zorder=4,
                    label="ground truth" if n == 0 else None)
        ax.legend(fontsize=8)
    for colour, j in zip(time_colours(picks.size), picks):
        ax.plot(Z[:, j].real, -Z[:, j].imag, "-o", ms=3, color=colour, zorder=3)
    ax.set(xlabel=r"Re$(Z)$ / $\Omega$", ylabel=r"$-$Im$(Z)$ / $\Omega$")
    ax.set_aspect("equal", adjustable="datalim")
    return np.asarray(times)[picks]


def _colorbar(fig, ax, times):
    norm = mpl.colors.Normalize(vmin=float(np.min(times)),
                                vmax=float(np.max(times)))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046).set_label("time / s")


def overview(time, voltage, current, fs, frequencies, Z, t_Z, Z_true=None):
    """Three columns: raw signals | their spectra | the impedance.

    ``Z`` is (n_freq, n_instants) reported at ``t_Z``; ``Z_true``, if given,
    is on the same instants. To show a system's own ground truth instead of
    an estimate, pass it as ``Z`` and leave ``Z_true`` out.
    """
    fig = plt.figure(figsize=(16, 6.5), layout="constrained")
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.0, 1.3])
    ax_v, ax_i = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_sv, ax_si = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[:, 2])

    envelope(ax_v, time, voltage, "C0")
    ax_v.set(ylabel="voltage / V", title="raw signals")
    envelope(ax_i, time, current, "C1")
    ax_i.set(xlabel="time / s", ylabel="current / A")

    spectrum(ax_sv, voltage, fs, frequencies, "C0")
    ax_sv.set(ylabel=r"$|V|$ / V", title="spectra")
    ax_sv.legend(fontsize=7, loc="lower left")
    spectrum(ax_si, current, fs, frequencies, "C1")
    ax_si.set(xlabel="frequency / Hz", ylabel=r"$|I|$ / A")

    shown = nyquist_vs_time(ax_z, np.atleast_2d(Z), t_Z, Z_true)
    ax_z.set_title("impedance" if Z_true is None else "estimated impedance")
    _colorbar(fig, ax_z, shown)
    return fig


def comparison(frequencies, results, n_instants=3):
    """Two panels: impedance against ground truth, and relative error.

    ``results`` is a list of ``(label, Z, t, Z_true)``, each ``Z`` and
    ``Z_true`` of shape (n_freq, len(t)).
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    colours = plt.get_cmap("tab10")(np.arange(max(len(results), 1)))

    label0, Z0, t0, truth0 = results[0]
    picks = np.unique(np.linspace(0, Z0.shape[1] - 1, n_instants).astype(int))
    for n, j in enumerate(picks):
        axs[0].plot(truth0[:, j].real, -truth0[:, j].imag, "-", color="black",
                    lw=2.2, alpha=0.45, zorder=2,
                    label="ground truth" if n == 0 else None)

    for colour, (label, Z, t, truth) in zip(colours, results):
        picks = np.unique(np.linspace(0, Z.shape[1] - 1, n_instants).astype(int))
        for n, j in enumerate(picks):
            axs[0].plot(Z[:, j].real, -Z[:, j].imag, "-o", ms=3, color=colour,
                        lw=1.0, label=label if n == 0 else None, zorder=3)
        err = np.abs(Z - truth) / np.abs(truth)
        axs[1].loglog(frequencies, np.median(err, axis=1), "-o", ms=4,
                      color=colour, label=label)
        if err.shape[1] > 2:
            axs[1].fill_between(frequencies,
                                np.percentile(err, 10, axis=1),
                                np.percentile(err, 90, axis=1),
                                color=colour, alpha=0.18, lw=0)

    axs[0].set(xlabel=r"Re$(Z)$ / $\Omega$", ylabel=r"$-$Im$(Z)$ / $\Omega$",
               title=f"{n_instants} instants through the record")
    axs[0].set_aspect("equal", adjustable="datalim")
    axs[0].legend(fontsize=8)
    axs[1].set(xlabel="frequency / Hz", ylabel="relative error",
               title="median over time, 10-90th percentile band")
    axs[1].grid(True, which="both", alpha=0.25)
    axs[1].legend(fontsize=8)
    return fig
