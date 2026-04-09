"""
Demo: Hantek DSO2D15 — enable wave generator, set a sine wave, capture a
sample buffer, and plot it with matplotlib.

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
# Configuration (tweak these for your experiment)
# ---------------------------------------------------------------------------
RESOURCE = None          # None → auto-discover, or e.g. "USB0::1183::..."
CHANNEL = 1              # Acquisition channel (1 or 2)

WAVE_TYPE = WaveType.SINE
WAVE_FREQ_HZ = 1000.0    # 1 kHz sine from the built-in generator
WAVE_AMP_VPP = 2.0       # 2 Vpp

V_DIV = 0.5              # Vertical scale: 0.5 V/div
TIME_DIV_S = 0.0005      # Horizontal scale: 500 µs/div
TRIGGER_LEVEL_V = 0.0    # Trigger at 0 V


def main() -> None:
    if RESOURCE:
        scope = DSO2D15(resource=RESOURCE)
        scope.connect()
    else:
        scope = DSO2D15.auto_discover()

    try:
        # -- 1. Basic oscilloscope setup -----------------------------------
        print("Configuring channel and timebase …")
        scope.setup_basic(
            channel=CHANNEL,
            v_div=V_DIV,
            s_div=TIME_DIV_S,
            coupling=Coupling.DC,
            trigger_level=TRIGGER_LEVEL_V,
        )

        # -- 2. Wave generator setup ---------------------------------------
        print(
            f"Setting wave generator: {WAVE_TYPE.value} @ {WAVE_FREQ_HZ} Hz, "
            f"{WAVE_AMP_VPP} Vpp …"
        )
        scope.set_wave_type(WAVE_TYPE)
        scope.set_wave_frequency(WAVE_FREQ_HZ)
        scope.set_wave_amplitude(WAVE_AMP_VPP)
        scope.enable_wave_output(True)

        # -- 3. Acquire ----------------------------------------------------
        print("Starting single-shot acquisition …")
        scope.stop()          # Ensure clean state
        time.sleep(0.2)       # Let the scope settle
        scope.single()

        # Wait until the scope has finished acquiring
        if not scope.wait_acquisition_done(timeout_s=10.0):
            print("WARNING: Acquisition timed out. Reading whatever is in the buffer.")
        else:
            print("Acquisition complete.")

        # -- 4. Read waveform data -----------------------------------------
        print("Reading waveform buffer …")
        voltages, y_incr, y_origin = scope.read_waveform(channel=CHANNEL)

        # Build the time axis from the timebase setting.
        num_samples = len(voltages)
        total_time = TIME_DIV_S * 10  # 10 horizontal divisions
        time_axis = np.linspace(0, total_time, num_samples)

        # -- 5. Plot -------------------------------------------------------
        print("Plotting …")
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis * 1e3, voltages, linewidth=0.8, color="cyan")
        plt.title(
            f"DSO2D15 — CH{CHANNEL}  |  "
            f"{WAVE_TYPE.value} {WAVE_FREQ_HZ} Hz, {WAVE_AMP_VPP} Vpp  |  "
            f"{V_DIV} V/div, {TIME_DIV_S * 1e6:.0f} µs/div"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # -- 6. Cleanup (wave gen off) -------------------------------------
        scope.enable_wave_output(False)
        print("Done. Wave generator output disabled.")
    finally:
        scope.close()


if __name__ == "__main__":
    main()
