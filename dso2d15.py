"""
Hantek DSO2D15 Oscilloscope SCPI Abstraction

Provides a high-level Python interface over the Hantek DSO2000 Series SCPI
command set using PyVISA with pyvisa-py backend.

On Windows the pyvisa-py USBTMC driver fails during capability negotiation
(control transfer errors with libusb0).  We monkey-patch the affected methods
before creating the session.

Key SCPI commands used (from the DSO2000 Series SCPI Programmers Manual):
  - :DDS:TYPE <sine|square|ramp|pulse|noise|dc>   — set wave-generator type
  - :DDS:FREQ <Hz>                                 — set wave-generator frequency
  - :DDS:AMP <Vpp>                                 — set wave-generator amplitude
  - :DDS:OFFSet <V>                                — set wave-generator DC offset
  - :DDS:SWITch <ON|OFF>                           — enable/disable wave generator
  - :CHANnel<N>:SCALe <V/div>                      — set channel vertical scale
  - :CHANnel<N>:OFFSet <V>                         — set channel vertical offset
  - :CHANnel<N>:COUPling <AC|DC>                   — set channel coupling
  - :TIMebase:SCALe <s/div>                        — set horizontal timebase
  - :WAVeform:DATA:ALL?                            — read captured waveform buffer
"""

import struct
import time
import warnings
from enum import Enum
from typing import Optional, Tuple, List

import pyvisa

# ---------------------------------------------------------------------------
# Patch pyvisa-py USBTMC for Windows (libusb0 control-transfer failures)
# ---------------------------------------------------------------------------
from pyvisa_py.protocols import usbtmc

_usbtmc_patched = False


def _patch_usbtmc_windows() -> None:
    """Patch pyvisa-py USBTMC to skip broken control transfers on Windows."""
    global _usbtmc_patched
    if _usbtmc_patched:
        return

    def _safe_get_capabilities(self):
        return usbtmc.UsbTmcCapabilities(
            usb488=False, ren_control=False, trigger=False
        )

    def _no_enable_remote_control(self):
        pass

    # Wrap write to handle claim_interface errors gracefully
    _original_raw_write = usbtmc.USBRaw.write

    def _safe_write(self, data):
        try:
            return _original_raw_write(self, data)
        except ValueError as e:
            if "claim_interface" in str(e):
                # Re-claim interface (may have been released)
                try:
                    import usb.util
                    usb.util.claim_interface(self.usb_dev, self.usb_intf)
                except Exception:
                    pass
                return _original_raw_write(self, data)
            raise

    usbtmc.USBTMC._get_capabilities = _safe_get_capabilities
    usbtmc.USBTMC._enable_remote_control = _no_enable_remote_control
    usbtmc.USBRaw.write = _safe_write
    _usbtmc_patched = True


# Apply the patch at import time
_patch_usbtmc_windows()
# ---------------------------------------------------------------------------


class WaveType(str, Enum):
    SINE = "SINE"
    SQUARE = "SQUARE"
    RAMP = "RAMP"
    PULSE = "PULSE"
    NOISE = "NOISE"
    DC = "DC"


class Coupling(str, Enum):
    AC = "AC"
    DC = "DC"
    GND = "GND"


class DSO2D15:
    """High-level driver for the Hantek DSO2D15 oscilloscope via SCPI."""

    # Default USBTMC resource string (VID:PID::serial::0::INSTR)
    DEFAULT_RESOURCE = "USB0::1183::20574::CN2317029041941::0::INSTR"
    DEFAULT_BACKEND = "@py"

    def __init__(
        self,
        resource: Optional[str] = None,
        backend: str = DEFAULT_BACKEND,
        timeout_ms: int = 5000,
    ):
        """
        Parameters
        ----------
        resource : str, optional
            VISA resource string.  Defaults to the DSO2D15 USB address.
        backend : str
            PyVISA backend.  ``"@py"`` uses pyvisa-py (pure Python USB).
        timeout_ms : int
            VISA timeout in milliseconds for read operations.
        """
        self._rm = pyvisa.ResourceManager(backend)
        self._resource = resource or self.DEFAULT_RESOURCE
        self._inst: Optional[pyvisa.resources.MessageBasedResource] = None
        self._timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the VISA session and verify communication."""
        self._inst = self._rm.open_resource(self._resource)
        self._inst.timeout = self._timeout_ms
        self._inst.read_termination = "\n"
        self._inst.write_termination = "\n"
        identity = self._inst.query("*IDN?").strip()
        print(f"Connected to: {identity}")

    def close(self) -> None:
        """Close the VISA session."""
        if self._inst is not None:
            self._inst.close()
            self._inst = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    @classmethod
    def auto_discover(cls, backend: str = DEFAULT_BACKEND) -> "DSO2D15":
        """Scan for the first USB instrument and return a connected DSO2D15."""
        rm = pyvisa.ResourceManager(backend)
        resources = rm.list_resources()
        for res in resources:
            if "USB" in res.upper():
                print(f"Found USB instrument: {res}")
                scope = cls(resource=res, backend=backend)
                scope.connect()
                return scope
        raise RuntimeError("No USB oscilloscope found.")

    # ------------------------------------------------------------------
    # Internal command helpers
    # ------------------------------------------------------------------
    def _write(self, cmd: str) -> None:
        assert self._inst is not None, "Not connected. Call .connect() first."
        self._inst.write(cmd)

    def _query(self, cmd: str) -> str:
        assert self._inst is not None, "Not connected. Call .connect() first."
        return self._inst.query(cmd).strip()

    # ------------------------------------------------------------------
    # Wave / DDS (Direct Digital Synthesis) commands
    # ------------------------------------------------------------------
    def set_wave_type(self, wave_type: WaveType) -> None:
        """Set the built-in wave generator waveform type."""
        self._write(f":DDS:TYPE {wave_type.value}")

    def set_wave_frequency(self, freq_hz: float) -> None:
        """Set the wave generator frequency in Hz."""
        self._write(f":DDS:FREQ {freq_hz}")

    def set_wave_amplitude(self, vpp: float) -> None:
        """Set the wave generator amplitude (peak-to-peak, in Volts)."""
        self._write(f":DDS:AMP {vpp}")

    def set_wave_offset(self, offset_v: float) -> None:
        """Set the wave generator DC offset in Volts."""
        self._write(f":DDS:OFFSet {offset_v}")

    def enable_wave_output(self, enabled: bool = True) -> None:
        """Turn the wave generator output ON or OFF."""
        self._write(f":DDS:SWITch {'ON' if enabled else 'OFF'}")

    def get_wave_output_state(self) -> str:
        """Query the current wave generator output state (ON/OFF)."""
        return self._query(":DDS:SWITch?")

    def get_wave_type(self) -> str:
        """Query the current wave generator type."""
        return self._query(":DDS:TYPE?")

    def get_wave_frequency(self) -> float:
        """Query the current wave generator frequency in Hz."""
        return float(self._query(":DDS:FREQ?"))

    # ------------------------------------------------------------------
    # Channel commands
    # ------------------------------------------------------------------
    def set_channel_scale(self, channel: int, volts_per_div: float) -> None:
        """Set vertical scale (V/div) for a channel (1 or 2)."""
        self._write(f":CHANnel{channel}:SCALe {volts_per_div}")

    def set_channel_offset(self, channel: int, offset_v: float) -> None:
        """Set vertical offset (in Volts) for a channel."""
        self._write(f":CHANnel{channel}:OFFSet {offset_v}")

    def set_channel_coupling(self, channel: int, coupling: Coupling) -> None:
        """Set channel coupling: AC, DC, or GND."""
        self._write(f":CHANnel{channel}:COUPling {coupling.value}")

    def enable_channel(self, channel: int, enabled: bool = True) -> None:
        """Enable or display a channel."""
        self._write(f":CHANnel{channel}:DISPlay {'ON' if enabled else 'OFF'}")

    # ------------------------------------------------------------------
    # Timebase commands
    # ------------------------------------------------------------------
    def set_timebase_scale(self, seconds_per_div: float) -> None:
        """Set horizontal timebase (seconds/div)."""
        self._write(f":TIMebase:SCALe {seconds_per_div}")

    def set_timebase_offset(self, seconds: float) -> None:
        """Set trigger delay / horizontal offset in seconds."""
        self._write(f":TIMebase:OFFSet {seconds}")

    # ------------------------------------------------------------------
    # Trigger commands
    # ------------------------------------------------------------------
    def set_trigger_edge_source(self, channel: int) -> None:
        """Set trigger source to CH1 or CH2 (edge trigger)."""
        self._write(f":TRIGger:EDGE:SOURce CHANnel{channel}")

    def set_trigger_level(self, level_v: float) -> None:
        """Set trigger voltage level."""
        self._write(f":TRIGger:LEVel {level_v}")

    # ------------------------------------------------------------------
    # Acquisition & waveform readback
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start acquisition (equivalent to pressing Run/Stop)."""
        self._write(":RUN")

    def stop(self) -> None:
        """Stop acquisition."""
        self._write(":STOP")

    def single(self) -> None:
        """Arm a single-shot acquisition."""
        self._write(":SINGle")

    def wait_acquisition_done(self, timeout_s: float = 10.0) -> bool:
        """
        Wait until the current acquisition is complete.

        Uses ``*OPC?`` (Operation Complete) for reliable synchronization.
        ``*OPC?`` blocks on the instrument side until all pending commands
        finish, then returns ``"1"``.  If the query times out, falls back
        to polling ``:TRIGger:STATus?``.

        Returns True if acquisition is confirmed done, False on timeout.
        """
        # Primary method: *OPC? is blocking on the instrument side
        try:
            old_timeout = self._inst.timeout
            self._inst.timeout = int(timeout_s * 1000)
            result = self._query("*OPC?").strip()
            self._inst.timeout = old_timeout
            return result == "1"
        except Exception:
            try:
                self._inst.timeout = old_timeout
            except Exception:
                pass

        # Fallback: poll trigger status
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                status = self._query(":TRIGger:STATus?").upper()
                if status in ("READY", "STOP"):
                    return True
            except Exception:
                pass
            time.sleep(0.05)

        return False

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate in Sa/s (if supported by firmware)."""
        self._write(f":ACQuire:SAMPling {sample_rate}")

    def read_waveform(
        self, channel: int = 1
    ) -> Tuple[List[float], float, float]:
        """
        Read the captured waveform buffer from the specified channel.

        Returns
        -------
        voltages : list of float
            Converted voltage values.
        y_increment : float
            Volts per ADC step.
        y_origin : float
            Reference voltage offset.
        """
        assert self._inst is not None, "Not connected."

        # Select the waveform source channel
        self._inst.write(f":WAVeform:SOURce CHANnel{channel}")

        # Send the query command
        self._inst.write(":WAVeform:DATA:ALL?")

        # Read raw bytes (IEEE 488.2 block with binary payload)
        raw_bytes = self._inst.read_raw()

        if not raw_bytes.startswith(b"#"):
            raise ValueError(
                f"Unexpected waveform header: {raw_bytes[:30]!r}"
            )

        # Parse IEEE 488.2 definite length block
        num_digits = int(chr(raw_bytes[1]))
        # The declared count includes the IEEE header itself, so subtract it
        ieee_header_len = 2 + num_digits
        declared_total = int(raw_bytes[2 : 2 + num_digits])
        expected_payload = declared_total - ieee_header_len

        # Extract payload (may be slightly short due to scope firmware quirks)
        actual_payload = raw_bytes[ieee_header_len:]

        # The DSO2000 payload starts with a 19-byte ASCII metadata block,
        # then binary 8-bit ADC samples. Example ASCII header:
        #   "0000040990000000099"  (sample_count + reserved fields)
        ASCII_HEADER_LEN = 19
        binary_data = actual_payload[ASCII_HEADER_LEN:]

        num_samples = len(binary_data)
        raw_values = struct.unpack(f"{num_samples}B", binary_data)

        # Compute scaling from channel settings
        try:
            v_div = float(self._query(f":CHANnel{channel}:SCALe?"))
            v_offset = float(self._query(f":CHANnel{channel}:OFFSet?"))
            # 10 vertical divisions, 8-bit ADC (256 levels)
            y_increment = (v_div * 10.0) / 256.0
            y_origin = v_offset
        except Exception:
            y_increment = 10.0 / 256.0
            y_origin = 0.0

        # Convert to voltages: ADC centered at 128, so 128 = 0V (plus offset)
        voltages = [(v if (v < 128) else (v - 256)) * y_increment + y_origin for v in raw_values]

        return list(voltages), y_increment, y_origin

    def _get_y_increment(self, channel: int) -> float:
        """Query volts per ADC step."""
        try:
            return float(self._query(f":WAVeform:YINCrement?"))
        except Exception:
            return 0.0

    def _get_y_origin(self, channel: int) -> float:
        """Query Y-origin voltage."""
        try:
            return float(self._query(f":WAVeform:YORigin?"))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Convenience: full setup in one call
    # ------------------------------------------------------------------
    def setup_basic(
        self,
        channel: int = 1,
        v_div: float = 1.0,
        s_div: float = 0.001,
        coupling: Coupling = Coupling.DC,
        trigger_level: float = 1.5,
        trigger_mode: str = "AUTO",
    ) -> None:
        """Apply a basic oscilloscope setup: channel scale, timebase, trigger."""
        self.enable_channel(channel)
        self.set_channel_scale(channel, v_div)
        self.set_channel_coupling(channel, coupling)
        self.set_timebase_scale(s_div)
        self.set_trigger_edge_source(channel)
        self.set_trigger_level(trigger_level)
        self.set_trigger_mode(trigger_mode)

    def set_trigger_mode(self, mode: str) -> None:
        """Set trigger mode: AUTO, NORMAL, or SINGLE."""
        self._write(f":TRIGger:MODE {mode}")
