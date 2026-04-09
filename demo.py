"""
Demo: Hantek DSO2D15 — enable wave generator, set a sine wave, capture a
sample buffer, and plot it with matplotlib.

Uses the correct `:DDS:SWITch` command to enable/disable the DDS output.

Usage
-----
    python demo.py

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
RESOURCE = None          # None → auto-discover
CHANNEL = 1
DISABLE_CH2 = True

WAVE_TYPE = WaveType.SINE
WAVE_FREQ_HZ = 1000.0
WAVE_AMP_VPP = 2.0

V_DIV = 0.5
TIME_DIV_S = 0.0005
TRIGGER_LEVEL_V = 0.0
ACQ_POINTS = 4000


def main() -> None:
    scope = DSO2D15.auto_discover() if not RESOURCE else DSO2D15(resource=RESOURCE)

    try:
        # -- 1. Oscilloscope setup -----------------------------------------
        print("Configuring scope …")
        scope.enable_channel(CHANNEL)
        if DISABLE_CH2:
            scope.enable_channel(2, False)
        scope.set_channel_scale(CHANNEL, V_DIV)
        scope.set_channel_coupling(CHANNEL, Coupling.DC)
        scope.set_timebase_scale(TIME_DIV_S)
        scope.set_acquisition_points(ACQ_POINTS)
        scope.set_trigger_edge_source(CHANNEL)
        scope.set_trigger_level(TRIGGER_LEVEL_V)

        # -- 2. DDS setup --------------------------------------------------
        print(f"Setting wave gen: {WAVE_TYPE.value} @ {WAVE_FREQ_HZ} Hz, {WAVE_AMP_VPP} Vpp …")
        scope.enable_wave_output(False)
        time.sleep(0.3)
        scope.set_wave_type(WAVE_TYPE)
        scope.set_wave_frequency(WAVE_FREQ_HZ)
        scope.set_wave_amplitude(WAVE_AMP_VPP)
        time.sleep(0.3)
        scope.enable_wave_output(True)
        time.sleep(1.0)  # Let DDS hardware stabilize

        # -- 3. Acquire ----------------------------------------------------
        print("Acquiring one fresh single-shot record …")

        # -- 4. Read & plot ------------------------------------------------
        print("Reading waveform …")
        voltages, _, _ = scope.capture_single_waveform(channel=CHANNEL, timeout_s=2.0)
        print(f"  {len(voltages)} samples, {min(voltages):.3f} V to {max(voltages):.3f} V")

        num_samples = len(voltages)
        try:
            sample_rate = scope.get_acquisition_sample_rate()
            total_time = num_samples / sample_rate if sample_rate > 0 else (TIME_DIV_S * 10)
            print(f"  Sample rate: {sample_rate:.3e} Sa/s, window: {total_time*1e3:.3f} ms")
        except Exception:
            total_time = TIME_DIV_S * 10
        time_axis = np.linspace(0, total_time, num_samples)

        plt.figure(figsize=(10, 5))
        plt.plot(time_axis * 1e3, voltages, linewidth=0.8, color="cyan")
        plt.title(
            f"DSO2D15 — CH{CHANNEL}  |  "
            f"{WAVE_TYPE.value} {WAVE_FREQ_HZ} Hz, {WAVE_AMP_VPP} Vpp"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # -- 5. Cleanup ----------------------------------------------------
        scope.enable_wave_output(False)
        print("Done. Wave gen disabled.")
    finally:
        scope.close()


if __name__ == "__main__":
    main()
