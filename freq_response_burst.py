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
    skip_zero_crossings: int = 2,
) -> tuple[float, np.ndarray]:
    """Measure RMS in a single time window after the burst excitation.

    Strategy:
      1. Find where the burst starts (first significant amplitude rise).
      2. Calculate burst end = burst_start + burst_cycles * period.
      3. Skip N positive-going zero crossings after burst end.
      4. Place the measurement window after those crossings.

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
    skip_zero_crossings : int
        Number of positive-going zero crossings to skip after burst end
        before starting the measurement window.

    Returns
    -------
    rms : float
        RMS amplitude in the measurement window.
    window_data : np.ndarray
        Raw voltage samples from the measurement window (for FFT/spectrogram).
    """
    arr = np.array(voltages, dtype=np.float64)
    n = len(arr)
    if n < 100:
        return 0.0, np.array([])

    dt_us = 1e6 / sample_rate  # µs per sample

    # Step 1: Compute envelope (RMS over a small sliding window) to find burst.
    env_win = max(4, int(2 / dt_us))  # ~2 µs sliding window
    envelope = np.zeros(n)
    for i in range(n):
        lo = max(0, i - env_win // 2)
        hi = min(n, i + env_win // 2)
        envelope[i] = np.sqrt(np.mean(np.square(arr[lo:hi])))

    # Step 2: Find the burst peak.
    peak_idx = int(np.argmax(envelope))
    peak_val = float(envelope[peak_idx])

    noise_floor = float(np.percentile(envelope, 20))
    burst_threshold = max(noise_floor * 3, peak_val * 0.10)

    # Step 3: Find burst START — scan backward from peak.
    burst_start_idx = int(peak_idx)
    while burst_start_idx > 0 and envelope[burst_start_idx - 1] > burst_threshold:
        burst_start_idx -= 1

    # Step 4: Estimate period from zero crossings in burst region.
    burst_region_end = min(n, burst_start_idx + int(200 / dt_us))
    crossings = []
    for i in range(burst_start_idx + 1, burst_region_end):
        if arr[i - 1] <= 0 and arr[i] > 0:
            crossings.append(i)

    if len(crossings) >= 2:
        estimated_period_us = float(np.mean(np.diff(crossings)) * dt_us)
    else:
        burst_end_idx_alt = int(peak_idx)
        while burst_end_idx_alt < n - 1 and envelope[burst_end_idx_alt + 1] > burst_threshold:
            burst_end_idx_alt += 1
        estimated_period_us = (burst_end_idx_alt - burst_start_idx) * dt_us / max(burst_cycles, 1)

    # Burst end = burst start + burst_cycles * period
    burst_end_idx = burst_start_idx + int(burst_cycles * estimated_period_us / dt_us)
    burst_end_idx = min(burst_end_idx, n - 1)

    # Step 5: Skip N positive-going zero crossings after burst end.
    skip_start = burst_end_idx + 1
    skip_limit = min(n, skip_start + int(500 / dt_us))  # search up to 500 µs
    skip_count = 0
    window_start_idx = skip_limit  # default: end of search range
    for i in range(skip_start + 1, skip_limit):
        if arr[i - 1] <= 0 and arr[i] > 0:
            skip_count += 1
            if skip_count >= skip_zero_crossings:
                window_start_idx = i
                break

    # Step 6: Extract measurement window.
    win_len = max(10, int(WINDOW1_DUR_US / dt_us))
    win_end_idx = min(n, window_start_idx + win_len)
    window_start_idx = max(0, min(window_start_idx, n - 1))
    win_end_idx = max(window_start_idx + 1, win_end_idx)

    window_data = arr[window_start_idx:win_end_idx]
    rms = calc_rms(window_data)

    # Debug on first capture
    global _debug_printed
    if not _debug_printed:
        _debug_printed = True
        t_bstart = burst_start_idx * dt_us - x_origin_s * 1e6
        t_bend = burst_end_idx * dt_us - x_origin_s * 1e6
        t_wstart = window_start_idx * dt_us - x_origin_s * 1e6
        t_wend = win_end_idx * dt_us - x_origin_s * 1e6
        print(f"\n  [debug] Peak envelope = {peak_val:.4f} V at sample {peak_idx}")
        print(f"  [debug] Zero crossings in burst: {len(crossings)}, "
              f"estimated period = {estimated_period_us:.2f} µs")
        print(f"  [debug] Burst: sample {burst_start_idx}–{burst_end_idx} "
              f"(t_rel = {t_bstart:.1f}–{t_bend:.1f} µs)")
        print(f"  [debug] Skipped {skip_count} zero crossings after burst end")
        print(f"  [debug] Window 1: samples {window_start_idx}–{win_end_idx} "
              f"({len(window_data)} pts, t_rel = {t_wstart:.1f}–{t_wend:.1f} µs)")

    return rms, window_data


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
        all_buffers: list[tuple[float, np.ndarray]] = []  # (exc_freq, full_voltages)
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
            rms, window_data = measure_window1(
                voltages, effective_sr, x_origin_s, BURST_CYCLES,
                skip_zero_crossings=2,
            )
            rms_values.append(rms)

            # Store full buffer for waveform viewer
            all_buffers.append((freq, np.array(voltages), effective_sr))

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

        # -- Plot: RMS + Spectrogram + Waveform Viewer ---------------------
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1.5], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        fig.suptitle(
            f"DSO2D15 Burst Frequency Response  —  "
            f"{BURST_CYCLES} cycle burst, {freqs[0]/1000:.0f}–{freqs[-1]/1000:.0f} kHz"
        )

        # Top: RMS amplitude vs excitation frequency
        (rms_line,) = ax1.plot(freqs_arr, rms_arr, linewidth=1.2, color="cyan", marker=".", markersize=3)
        rms_peak_vline = ax1.axvline(freqs_arr[max_idx], color="gray", linewidth=0.5,
                                     linestyle="--")
        # Slider indicator on RMS plot
        (slider_vline,) = ax1.plot([freqs_arr[0], freqs_arr[0]], [0, np.max(rms_arr) * 1.1],
                                   color="red", linewidth=1.5, linestyle="-", alpha=0.7)
        ax1.set_ylabel("RMS Amplitude (V)")
        ax1.grid(True, alpha=0.3)
        ax1.legend([rms_line], ["RMS Response"], fontsize=8)

        # Middle: Spectrogram
        im = ax2.pcolormesh(
            freqs_arr, common_freqs / 1000, intensity_db.T,
            shading="auto", cmap="viridis", vmin=-60, vmax=0,
        )
        ax2.set_ylabel("Response Frequency (kHz)")
        ax2.set_title("Response Spectrum (dB, relative)")
        fig.colorbar(im, ax=ax2, label="Intensity (dB)")

        # Bottom: Waveform viewer (initially shows first frequency)
        init_freq, init_buf, init_sr = all_buffers[0]
        init_sr = float(init_sr) if init_sr > 0 else 12.5e6
        init_dt_us = 1e6 / init_sr
        init_time_us = np.arange(len(init_buf)) * init_dt_us
        (wave_line,) = ax3.plot(init_time_us, init_buf, linewidth=1.5, color="black")
        wave_title = ax3.set_title(f"Waveform @ {init_freq/1000:.1f} kHz  |  {len(init_buf)} samples  |  {init_sr/1e6:.1f} MSa/s")
        ax3.set_xlabel("Time (µs)")
        ax3.set_ylabel("Voltage (V)")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, init_time_us[-1])

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
            _, buf, sr = all_buffers[idx]
            sr = float(sr) if sr > 0 else 12.5e6
            dt_us = 1e6 / sr
            time_us = np.arange(len(buf)) * dt_us

            wave_line.set_data(time_us, buf)
            ax3.set_xlim(0, time_us[-1])

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
