"""
Frequency Response Analyzer — Hantek DSO2D15

Sweeps the DDS signal generator from 40 kHz to 200 kHz in 2 kHz steps,
captures the waveform at each frequency, measures the peak-to-peak
amplitude, and plots the frequency response curve.

Usage
-----
    python freq_response.py

Requirements
------------
    pip install pyvisa pyvisa-py matplotlib numpy
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from dso2d15 import DSO2D15, WaveType, Coupling

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESOURCE = None          # None -> auto-discover
CHANNEL = 1

FREQ_START_HZ = 40_000
FREQ_STOP_HZ = 200_000
FREQ_STEP_HZ = 2_000

DDS_AMP_VPP = 2.0        # DDS output amplitude (fixed)
DDS_OFFSET_V = 0.0

V_DIV = 0.5              # Scope vertical scale
TIME_DIV_S = 50e-6       # Timebase — enough cycles at each frequency
TRIGGER_LEVEL_V = 0.0
SETTLE_S = 0.5           # Wait after frequency change before capture


def measure_amplitude(voltages: list[float]) -> float:
    """Return peak-to-peak amplitude, ignoring edge transients.

    Discards the first and last 10 % of samples to avoid trigger-artifact
    distortion, then computes Vpp from the central 80 %.
    """
    n = len(voltages)
    if n < 20:
        return max(voltages) - min(voltages)

    margin = n // 10
    core = voltages[margin : n - margin]
    return max(core) - min(core)


def fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable HH:MM:SS string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    scope = DSO2D15.auto_discover() if not RESOURCE else DSO2D15(resource=RESOURCE)

    try:
        # -- Scope setup ---------------------------------------------------
        print("Configuring oscilloscope …")
        scope.enable_channel(CHANNEL)
        scope.set_channel_scale(CHANNEL, V_DIV)
        scope.set_channel_coupling(CHANNEL, Coupling.DC)
        scope.set_timebase_scale(TIME_DIV_S)
        scope.set_trigger_edge_source(CHANNEL)
        scope.set_trigger_level(TRIGGER_LEVEL_V)
        scope.set_trigger_mode("AUTO")

        # -- DDS setup (fixed amplitude, variable frequency) ---------------
        print("Initializing DDS …")
        scope.enable_wave_output(False)
        time.sleep(0.2)
        scope.set_wave_type(WaveType.SINE)
        scope.set_wave_amplitude(DDS_AMP_VPP)
        scope.set_wave_offset(DDS_OFFSET_V)
        scope.enable_wave_output(True)
        time.sleep(0.5)

        # -- Build frequency list ------------------------------------------
        freqs = list(range(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ, FREQ_STEP_HZ))
        n_steps = len(freqs)
        print(f"\nSweeping {n_steps} points: "
              f"{freqs[0]/1000:.0f} kHz → {freqs[-1]/1000:.0f} kHz "
              f"(step {FREQ_STEP_HZ/1000:.0f} kHz)\n")

        amplitudes: list[float] = []
        t_start = time.time()

        # -- Sweep loop ----------------------------------------------------
        for i, freq in enumerate(freqs, 1):
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (n_steps - i) if i > 1 else 0

            # Set frequency and let DDS settle
            scope.set_wave_frequency(freq)
            time.sleep(SETTLE_S)

            # Force a fresh single-shot acquisition
            scope._write(":TRIGger:MODE AUTO")
            scope._write(":SINGle")

            # Wait for acquisition to complete
            scope.wait_acquisition_done(timeout_s=2.0)

            # Small delay to ensure buffer is ready
            time.sleep(0.1)

            # Read waveform
            voltages, _, _ = scope.read_waveform(channel=CHANNEL)

            # Measure
            vpp = measure_amplitude(voltages)
            amplitudes.append(vpp)

            # Progress bar
            pct = i / n_steps * 100
            bar_w = 30
            filled = int(bar_w * i / n_steps)
            bar = "█" * filled + "░" * (bar_w - filled)
            freq_k = freq / 1000
            print(f"\r  [{bar}] {pct:5.1f}%  |  "
                  f"{freq_k:7.1f} kHz  →  Vpp = {vpp:.4f} V  |  "
                  f"ETA {fmt_duration(eta)}", end="", flush=True)

        print()  # newline after progress bar

        total_time = time.time() - t_start
        print(f"\nSweep complete in {fmt_duration(total_time)}.")

        # -- Analysis ------------------------------------------------------
        vpp_arr = np.array(amplitudes)
        freqs_arr = np.array(freqs) / 1000  # kHz
        ref_vpp = vpp_arr[0]
        if ref_vpp > 0:
            db_vals = 20 * np.log10(vpp_arr / ref_vpp)
        else:
            db_vals = np.full_like(vpp_arr, -np.inf)

        max_idx = np.argmax(vpp_arr)
        print(f"\n  Max amplitude : {vpp_arr[max_idx]:.4f} Vpp "
              f"at {freqs_arr[max_idx]:.1f} kHz")
        print(f"  Min amplitude : {np.min(vpp_arr):.4f} Vpp "
              f"at {freqs_arr[np.argmin(vpp_arr)]:.1f} kHz")

        # -- Plot: Amplitude vs Frequency ----------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [1, 1]})
        fig.suptitle(
            f"DSO2D15 Frequency Response  —  "
            f"DDS Sine {DDS_AMP_VPP} Vpp, {freqs[0]/1000:.0f}–{freqs[-1]/1000:.0f} kHz"
        )

        # Linear plot
        ax1.plot(freqs_arr, vpp_arr, linewidth=1.2, color="cyan", marker=".", markersize=3)
        ax1.set_ylabel("Amplitude (Vpp)")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(freqs_arr[max_idx], color="gray", linewidth=0.5,
                    linestyle="--", label=f"Peak @ {freqs_arr[max_idx]:.1f} kHz")
        ax1.legend(fontsize=8)

        # dB plot (relative to first point)
        ax2.plot(freqs_arr, db_vals, linewidth=1.2, color="orange", marker=".", markersize=3)
        ax2.set_xlabel("Frequency (kHz)")
        ax2.set_ylabel("Amplitude (dB, rel.)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        plt.tight_layout()
        plt.show()

        # -- Cleanup -------------------------------------------------------
        scope.enable_wave_output(False)
        print("\nDDS output disabled. Done.")

    finally:
        scope.close()


if __name__ == "__main__":
    main()
