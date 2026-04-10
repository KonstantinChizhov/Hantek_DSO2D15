"""
Frequency Response Analyzer — Burst Excitation Mode (Hantek DSO2D15)

Sweeps the DDS signal generator from 40 kHz to 200 kHz in 2 kHz steps.
At each frequency, the oscilloscope's built-in burst mode generates a
short burst (1-3 cycles). The scope triggers on the burst edge and
measures RMS amplitude in a single window that starts after 2 zero
crossings following the burst end. A spectrogram shows the frequency
content of each response window.

SCPI burst commands used (DSO2000 manual):
  :DDS:BURSt:SWITch <ON|OFF>   — enable/disable burst mode
  :DDS:BURSt:TYPE <NORM|INF>   — burst type (Normal = finite cycles)
  :DDS:BURSt:CNT <n>           — number of cycles per burst
  :DDS:BURSt:TRIGger           — software trigger to fire one burst

Usage
-----
    python freq_response_burst.py

Requirements
------------
    pip install pyvisa pyvisa-py matplotlib numpy
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dso2d15 import DSO2D15, WaveType, Coupling

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESOURCE = None          # None -> auto-discover
CHANNEL = 1
DISABLE_CH2 = True

FREQ_START_HZ = 40_000
FREQ_STOP_HZ = 200_000
FREQ_STEP_HZ = 2_000

BURST_CYCLES = 2         # Number of cycles per burst (1-3 recommended)
DDS_AMP_VPP = 5.0        # DDS output amplitude (Vpp)
DDS_OFFSET_V = 0.0

V_DIV = 0.5              # Scope vertical scale
TRIGGER_LEVEL_V = 0.1    # Trigger threshold to catch burst edge
ACQ_POINTS = 4000        # Memory depth per capture

# RMS measurement window (after burst end + 2 zero crossings)
WINDOW1_DUR_US = 50      # Duration of the measurement window (µs)

SETTLE_S = 0.05          # Pause after frequency change


def calc_rms(samples: np.ndarray) -> float:
    """Calculate RMS amplitude of a sample array."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def measure_window1(
    voltages: list[float],
    sample_rate: float,
    x_origin_s: float,
    burst_cycles: int,
    excitation_freq_hz: float,
) -> tuple[float, np.ndarray, int, int]:
    """Measure RMS in a single time window after the burst excitation.

    Strategy (deterministic, frequency-anchored):
      1. Find burst PEAK index (stable anchor — argmax is deterministic).
      2. Calculate exact burst duration from known frequency: duration = cycles / freq.
      3. Find burst start = peak - (burst_cycles/2) * period (peak ≈ center of burst).
      4. burst_end = burst_start + burst_duration.
      5. Place measurement window at burst_end + gap.

    Parameters
    ----------
    voltages : list
        Captured waveform samples.
    sample_rate : float
        Sampling rate in Sa/s.
    x_origin_s : float
        Time offset of sample[0] relative to trigger (from :WAVeform:XORigin?).
    burst_cycles : int
        Number of cycles in the excitation burst.
    excitation_freq_hz : float
        The known DDS excitation frequency in Hz (set before capture).

    Returns
    -------
    rms : float
        RMS amplitude in the measurement window.
    window_data : np.ndarray
        Raw voltage samples from the measurement window (for FFT/spectrogram).
    win_start_idx : int
        Sample index where the measurement window starts.
    win_end_idx : int
        Sample index where the measurement window ends.
    """
    arr = np.array(voltages, dtype=np.float64)
    n = len(arr)
    if n < 100:
        return 0.0, np.array([]), 0, 0

    dt_us = 1e6 / sample_rate  # µs per sample

    # ---- Step 1: Compute envelope (RMS over sliding window) ----------------
    env_win_samples = max(8, int(5 / dt_us))  # ~5 µs sliding window
    half_env = env_win_samples // 2
    envelope = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half_env)
        hi = min(n, i + half_env)
        envelope[i] = np.sqrt(np.mean(np.square(arr[lo:hi])))

    # ---- Step 2: Find burst PEAK (most stable feature) ---------------------
    peak_idx = int(np.argmax(envelope))
    peak_val = float(envelope[peak_idx])

    if peak_val < 1e-6:
        # No signal detected
        win_len = max(10, int(WINDOW1_DUR_US / dt_us))
        return 0.0, np.zeros(win_len), n // 2, n // 2 + win_len

    # ---- Step 3: Calculate burst timing from known frequency ---------------
    # Period is known exactly from the DDS frequency
    period_us = 1e6 / excitation_freq_hz

    # For a burst_cycles burst, the burst duration is exactly:
    burst_duration_us = burst_cycles * period_us

    # The peak of the envelope should be approximately in the center of the burst.
    # So: burst_start = peak - (burst_cycles / 2) * period
    #     burst_end   = burst_start + burst_duration
    half_burst_us = burst_duration_us / 2.0
    burst_center_us = peak_idx * dt_us
    burst_start_us = burst_center_us - half_burst_us
    burst_end_us = burst_start_us + burst_duration_us

    burst_end_idx = int(burst_end_us / dt_us)

    # ---- Step 4: Place measurement window after burst + gap ----------------
    # Gap of ~1 period ensures we're past any ringing/settling from the burst
    gap_us = max(5.0, period_us * 0.8)
    window_start_us = burst_end_us + gap_us
    window_start_idx = int(window_start_us / dt_us)

    # Bounds checking
    win_len = max(10, int(WINDOW1_DUR_US / dt_us))
    if window_start_idx + win_len > n:
        window_start_idx = max(0, n - win_len)
    if window_start_idx < 0:
        window_start_idx = 0

    win_end_idx = min(n, window_start_idx + win_len)
    win_end_idx = max(window_start_idx + 1, win_end_idx)

    window_data = arr[window_start_idx:win_end_idx]
    rms = calc_rms(window_data)

    return rms, window_data, window_start_idx, win_end_idx


_debug_printed = False


def fmt_duration(seconds: float) -> str:
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
        if DISABLE_CH2:
            scope.enable_channel(2, False)
        scope.set_channel_scale(CHANNEL, V_DIV)
        scope.set_channel_coupling(CHANNEL, Coupling.DC)
        scope.set_channel_offset(CHANNEL, 0.0)
        scope.set_acquisition_points(ACQ_POINTS)

        # Edge trigger on the burst rising edge, normal sweep
        scope.set_trigger_edge_source(CHANNEL)
        scope.set_trigger_level(TRIGGER_LEVEL_V)
        scope.set_trigger_sweep("NORMal")

        # -- DDS burst setup (done once) -----------------------------------
        print("Initializing DDS burst mode …")
        scope.enable_wave_output(False)
        time.sleep(0.2)
        scope.set_wave_type(WaveType.SINE)
        scope.set_wave_amplitude(DDS_AMP_VPP)
        scope.set_wave_offset(DDS_OFFSET_V)

        # Configure burst mode
        scope.set_burst_cycle_count(BURST_CYCLES)
        scope.set_burst_type("NORM")
        scope.enable_burst_mode(True)
        scope.enable_wave_output(True)   # DDS output enabled
        time.sleep(0.5)

        # -- Build frequency list ------------------------------------------
        freqs = list(range(FREQ_START_HZ, FREQ_STOP_HZ + FREQ_STEP_HZ, FREQ_STEP_HZ))
        n_steps = len(freqs)
        print(f"\nSweeping {n_steps} points: "
              f"{freqs[0]/1000:.0f} kHz → {freqs[-1]/1000:.0f} kHz "
              f"(step {FREQ_STEP_HZ/1000:.0f} kHz)")
        print(f"Burst: {BURST_CYCLES} cycle(s), amplitude {DDS_AMP_VPP} Vpp")
        print(f"RMS Window: {WINDOW1_DUR_US} µs starting after 2 zero crossings post-burst\n")

        rms_values: list[float] = []
        response_spectra: list[tuple[float, np.ndarray, np.ndarray]] = []  # (exc_freq, fft_freqs, magnitude)
        # Store: (freq, voltages, sample_rate, win_start_idx, win_end_idx)
        all_buffers: list[tuple[float, np.ndarray, float, int, int]] = []
        t_start = time.time()

        # -- Sweep loop ----------------------------------------------------
        for i, freq in enumerate(freqs, 1):
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (n_steps - i) if i > 1 else 0

            # Set frequency (DDS stays enabled in burst mode)
            scope.set_wave_frequency(freq)
            time.sleep(SETTLE_S)

            # Drain any pending replies before starting fresh
            scope.clear_buffer()

            # 1) Arm the scope in single-shot mode (scope waits for trigger)
            scope.set_trigger_sweep("SINGle")
            time.sleep(0.01)

            # 2) Fire one burst — this is what the scope triggers on
            scope.trigger_burst()

            # 3) Wait for the scope to capture the burst (no force trigger)
            voltages, sample_rate, _ = scope.capture_triggered_waveform(
                channel=CHANNEL,
                timeout_s=2.0,
                retries=2,
                read_retries_per_capture=2,
                read_timeout_ms=5000,
            )

            # Query timing info
            x_origin_s = scope.get_waveform_x_origin()
            x_increment_s = scope.get_waveform_x_increment()
            effective_sr = 1.0 / x_increment_s if x_increment_s > 0 else sample_rate

            # Measure RMS and extract response window data
            rms, window_data, win_start, win_end = measure_window1(
                voltages, effective_sr, x_origin_s, BURST_CYCLES,
                excitation_freq_hz=freq,
            )
            rms_values.append(rms)

            # Store full buffer for waveform viewer with window indices
            all_buffers.append((freq, np.array(voltages), effective_sr, win_start, win_end))

            # Compute spectrum of the response window for spectrogram
            if len(window_data) > 0:
                n_pts = len(window_data)
                # Apply Hanning window to reduce spectral leakage
                windowed = window_data * np.hanning(n_pts)
                fft_vals = np.fft.rfft(windowed)
                fft_freqs = np.fft.rfftfreq(n_pts, d=1.0 / effective_sr)
                magnitude = np.abs(fft_vals) / n_pts
                response_spectra.append((freq, fft_freqs, magnitude))
            else:
                response_spectra.append((freq, np.array([]), np.array([])))

            # Progress
            pct = i / n_steps * 100
            bar_w = 30
            filled = int(bar_w * i / n_steps)
            bar = "█" * filled + "░" * (bar_w - filled)
            freq_k = freq / 1000
            print(f"\r  [{bar}] {pct:5.1f}%  |  "
                  f"{freq_k:7.1f} kHz  →  RMS = {rms:.4f} V  |  "
                  f"ETA {fmt_duration(eta)}", end="", flush=True)

        print()

        total_time = time.time() - t_start
        print(f"\nSweep complete in {fmt_duration(total_time)}.")

        # -- Analysis ------------------------------------------------------
        rms_arr = np.array(rms_values)
        freqs_arr = np.array(freqs) / 1000  # kHz

        max_idx = np.argmax(rms_arr)
        print(f"\n  Max RMS: {rms_arr[max_idx]:.4f} V "
              f"at {freqs_arr[max_idx]:.1f} kHz")
        print(f"  Min RMS: {np.min(rms_arr):.4f} V "
              f"at {freqs_arr[np.argmin(rms_arr)]:.1f} kHz")

        # -- Build spectrogram data ----------------------------------------
        max_fft_freq = 500_000  # 500 kHz max display
        common_freqs = np.linspace(0, max_fft_freq, 500)  # 500 bins

        # Build 2D intensity matrix: rows = excitation freq, cols = response freq
        intensity = np.zeros((n_steps, len(common_freqs)))
        for j, (exc_freq, fft_freqs, magnitude) in enumerate(response_spectra):
            if len(fft_freqs) == 0:
                continue
            mask = fft_freqs <= max_fft_freq
            interp_vals = np.interp(
                common_freqs,
                fft_freqs[mask],
                magnitude[mask],
                left=0, right=0,
            )
            intensity[j, :] = interp_vals

        # Normalize: convert to dB relative to global max
        global_max = np.max(intensity)
        if global_max > 0:
            intensity_db = 20 * np.log10(intensity / global_max + 1e-12)
            intensity_db = np.clip(intensity_db, -60, 0)
        else:
            intensity_db = np.zeros_like(intensity) - 60

        # -- Extract dominant frequencies from spectra ---------------------
        # For each excitation frequency, find the strongest response frequency
        dom_freq1_khz = np.zeros(n_steps)

        for j, (exc_freq, fft_freqs, magnitude) in enumerate(response_spectra):
            if len(fft_freqs) < 3:
                dom_freq1_khz[j] = np.nan
                continue
            # Find index of the largest peak (skip DC at index 0)
            top1 = np.argmax(magnitude[1:]) + 1
            dom_freq1_khz[j] = fft_freqs[top1] / 1000

        # -- Plot: RMS + Spectrogram + Waveform Viewer ---------------------
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1.5], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax1r = ax1.twinx()  # secondary y-axis for dominant frequencies
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        fig.suptitle(
            f"DSO2D15 Burst Frequency Response  —  "
            f"{BURST_CYCLES} cycle burst, {freqs[0]/1000:.0f}–{freqs[-1]/1000:.0f} kHz"
        )

        # Top: RMS amplitude + dominant frequencies on secondary axis
        (rms_line,) = ax1.plot(freqs_arr, rms_arr, linewidth=1.2, color="cyan",
                               marker=".", markersize=3, label="RMS Response")
        rms_peak_vline = ax1.axvline(freqs_arr[max_idx], color="gray", linewidth=0.5,
                                     linestyle="--")
        # Slider indicator on RMS plot
        (slider_vline,) = ax1.plot([freqs_arr[0], freqs_arr[0]],
                                   [0, np.max(rms_arr) * 1.1],
                                   color="red", linewidth=1.5, linestyle="-", alpha=0.7)

        (dom1_line,) = ax1r.plot(freqs_arr, dom_freq1_khz, linewidth=1.0,
                                 color="orange", marker="x", markersize=4,
                                 alpha=0.9, label="Dominant Freq")

        ax1.set_ylabel("RMS Amplitude (V)")
        ax1.set_xlabel("Excitation Frequency (kHz)")
        ax1.grid(True, alpha=0.3)

        ydom_max = np.nanmax(dom_freq1_khz)
        if np.isfinite(ydom_max) and ydom_max > 0:
            ax1r.set_ylim(0, ydom_max * 1.2)
        ax1r.set_ylabel("Dominant Response Freq (kHz)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

        # Middle: Spectrogram
        im = ax2.pcolormesh(
            freqs_arr, common_freqs / 1000, intensity_db.T,
            shading="auto", cmap="viridis", vmin=-60, vmax=0,
        )
        ax2.set_ylabel("Response Frequency (kHz)")
        ax2.set_title("Response Spectrum (dB, relative)")
        fig.colorbar(im, ax=ax2, label="Intensity (dB)")

        # Bottom: Waveform viewer with window markers
        init_freq, init_buf, init_sr, init_ws, init_we = all_buffers[0]
        init_sr = float(init_sr) if init_sr > 0 else 12.5e6
        init_dt_us = 1e6 / init_sr
        init_time_us = np.arange(len(init_buf)) * init_dt_us
        (wave_line,) = ax3.plot(init_time_us, init_buf, linewidth=0.8, color="black")

        # Vertical lines marking the measurement window
        win_vline1 = ax3.axvline(init_ws * init_dt_us, color="lime",
                                 linewidth=1.5, linestyle="-", alpha=0.7,
                                 label="Window Start")
        win_vline2 = ax3.axvline(init_we * init_dt_us, color="yellow",
                                 linewidth=1.5, linestyle="-", alpha=0.7,
                                 label="Window End")
        # Shaded region for the window
        win_fill = ax3.axvspan(init_ws * init_dt_us, init_we * init_dt_us,
                               alpha=0.08, color="lime")
        wave_title = ax3.set_title(
            f"Waveform @ {init_freq/1000:.1f} kHz  |  "
            f"{len(init_buf)} samples  |  {init_sr/1e6:.1f} MSa/s")
        ax3.set_xlabel("Time (µs)")
        ax3.set_ylabel("Voltage (V)")
        ax3.set_ylim(-2.0, 2.0)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, init_time_us[-1])
        ax3.legend(fontsize=8)

        # Slider at the bottom
        slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
        freq_slider = Slider(
            ax=slider_ax,
            label="Excitation Freq (kHz)",
            valmin=freqs_arr[0],
            valmax=freqs_arr[-1],
            valinit=freqs_arr[0],
            valstep=freqs_arr,
        )

        # Build lookup: freq -> buffer index
        freq_to_idx = {f: i for i, f in enumerate(freqs)}

        def update_waveform(val):
            """Update waveform plot when slider changes."""
            sel_freq_khz = freq_slider.val
            sel_freq_hz = int(sel_freq_khz * 1000)
            idx = freq_to_idx.get(sel_freq_hz)
            if idx is None:
                return
            _, buf, sr, ws, we = all_buffers[idx]
            sr = float(sr) if sr > 0 else 12.5e6
            dt_us = 1e6 / sr
            time_us = np.arange(len(buf)) * dt_us

            wave_line.set_data(time_us, buf)
            ax3.set_xlim(0, time_us[-1])

            # Update window markers
            win_vline1.set_xdata([ws * dt_us, ws * dt_us])
            win_vline2.set_xdata([we * dt_us, we * dt_us])
            # Update shaded region (Rectangle patch from axvspan)
            win_fill.set_xy((ws * dt_us, -2.0))
            win_fill.set_width((we - ws) * dt_us)
            win_fill.set_height(4.0)

            # Update slider indicator on RMS plot
            slider_vline.set_xdata([sel_freq_khz, sel_freq_khz])

            # Update title
            wave_title.set_text(
                f"Waveform @ {sel_freq_khz:.1f} kHz  |  "
                f"{len(buf)} samples  |  {sr/1e6:.1f} MSa/s"
            )
            fig.canvas.draw_idle()

        freq_slider.on_changed(update_waveform)

        plt.show()

        # -- Cleanup -------------------------------------------------------
        scope.enable_burst_mode(False)
        scope.enable_wave_output(False)
        print("\nDDS burst disabled. Done.")

    finally:
        scope.close()


if __name__ == "__main__":
    main()
