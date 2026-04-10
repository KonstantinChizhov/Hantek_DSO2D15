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
    # DDS Burst commands
    # ------------------------------------------------------------------
    def set_burst_cycle_count(self, count: int) -> None:
        """Set the number of cycles per burst."""
        self._write(f":DDS:BURSt:CNT {int(count)}")

    def get_burst_cycle_count(self) -> int:
        """Query the current burst cycle count."""
        return int(self._query(":DDS:BURSt:CNT?"))

    def set_burst_type(self, burst_type: str) -> None:
        """Set burst type: NORM (finite), INF (infinite), or GATE (gated)."""
        self._write(f":DDS:BURSt:TYPE {burst_type}")

    def get_burst_type(self) -> str:
        """Query the current burst type."""
        return self._query(":DDS:BURSt:TYPE?")

    def trigger_burst(self) -> None:
        """Fire a single software-triggered burst."""
        self._write(":DDS:BURSt:TRIGger")

    def enable_burst_mode(self, enabled: bool = True) -> None:
        """Enable or disable DDS burst mode."""
        self._write(f":DDS:BURSt:SWITch {'ON' if enabled else 'OFF'}")

    def get_burst_mode_state(self) -> str:
        """Query DDS burst mode state."""
        return self._query(":DDS:BURSt:SWITch?")

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
        Wait until a trigger event is reported.

        Per the DSO2000 SCPI manual, ``:TRIGger:STATus?`` returns ``TRIGed`` or
        ``NOTRIG``. For single-shot capture we treat ``TRIGed`` as completion.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                status = self._query(":TRIGger:STATus?").upper()
                if "TRIG" in status and "NOTRIG" not in status:
                    return True
            except Exception:
                pass
            time.sleep(0.05)

        return False

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate in Sa/s (if supported by firmware)."""
        self._write(f":ACQuire:SAMPling {sample_rate}")

    def set_acquisition_points(self, points: int) -> None:
        """Set acquisition memory depth (e.g., 4000, 40000, 400000...)."""
        self._write(f":ACQuire:POINts {int(points)}")

    def get_acquisition_points(self) -> int:
        """Query acquisition memory depth."""
        return int(float(self._query(":ACQuire:POINts?")))

    def get_acquisition_sample_rate(self) -> float:
        """Query current acquisition sampling rate in Sa/s."""
        return float(self._query(":ACQuire:SRATe?"))

    def get_waveform_x_origin(self) -> float:
        """Query the time offset (seconds) of the first sample relative to trigger.

        Returns a negative value when pre-trigger data is present in the buffer.
        A value of 0 means the trigger is at sample index 0.
        """
        return float(self._query(":WAVeform:XORigin?"))

    def get_waveform_x_increment(self) -> float:
        """Query the time interval (seconds) between consecutive samples."""
        return float(self._query(":WAVeform:XINCrement?"))

    def set_acquisition_state(self, running: bool) -> None:
        """Explicitly set acquisition state without relying on front-panel toggle logic."""
        self._write(f":ACQuire:STATE {'ON' if running else 'OFF'}")

    def clear_buffer(self) -> None:
        """Flush any pending data in the VISA input buffer."""
        if self._inst is None:
            return
        try:
            old_timeout = self._inst.timeout
            self._inst.timeout = 20
            while True:
                self._inst.read_raw()
        except Exception:
            try:
                self._inst.timeout = old_timeout
            except Exception:
                pass

    def capture_single_waveform(
        self,
        channel: int = 1,
        timeout_s: float = 2.0,
        retries: int = 1,
        min_samples: int = 1000,
        read_retries_per_capture: int = 1,
        read_timeout_ms: int = 3000,
    ) -> Tuple[List[float], float, float]:
        """
        Capture and read one fresh waveform record.

        This sequence avoids stale records by forcing acquisition into a known
        state, arming a single-shot capture, waiting for completion, and only
        then reading the waveform data.
        """
        assert self._inst is not None, "Not connected."

        # Drain any pending reply fragments before starting a new capture.
        self.clear_buffer()

        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, retries) + 1):
            # SCPI-manual-compliant single shot flow:
            #   1) select single trigger sweep
            #   2) force one trigger to ensure capture from STOP state
            self.set_trigger_sweep("SINGle")
            time.sleep(0.01)
            self.force_trigger()

            done = self.wait_acquisition_done(timeout_s=timeout_s)
            if not done:
                warnings.warn(
                    f"Single acquisition did not complete within {timeout_s:.2f}s; "
                    "reading last available waveform record.",
                    RuntimeWarning,
                )

            for read_try in range(1, max(1, read_retries_per_capture) + 1):
                try:
                    voltages, y_inc, y_org = self.read_waveform(
                        channel=channel,
                        read_timeout_ms=read_timeout_ms,
                    )
                    if len(voltages) >= max(1, min_samples):
                        return voltages, y_inc, y_org
                    raise ValueError(
                        f"Waveform payload too short: {len(voltages)} samples "
                        f"(expected >= {max(1, min_samples)})."
                    )
                except Exception as exc:
                    last_error = exc
                    if read_try < max(1, read_retries_per_capture):
                        # Re-read the same frozen capture before forcing a new one.
                        time.sleep(0.05)
                        continue

            if attempt < max(1, retries):
                # Start a fresh acquisition cycle after read retries are exhausted.
                self.clear_buffer()
                time.sleep(0.05)
                continue
            break

        raise RuntimeError(
            f"Failed to capture non-empty waveform after {max(1, retries)} acquisition "
            f"attempt(s) and {max(1, read_retries_per_capture)} read retry(ies) each."
        ) from last_error

    def capture_triggered_waveform(
        self,
        channel: int = 1,
        timeout_s: float = 2.0,
        retries: int = 2,
        min_samples: int = 100,
        read_retries_per_capture: int = 2,
        read_timeout_ms: int = 5000,
    ) -> Tuple[List[float], float, float]:
        """
        Arm a single-shot capture and wait for an **external trigger** (e.g. DDS burst).

        Unlike :meth:`capture_single_waveform`, this does **not** call
        ``:TRIGger:FORCe`` — it arms the scope in single-shot mode and waits
        for an incoming trigger event from the signal itself.

        Typical usage for burst capture::

            scope.set_trigger_sweep("SINGle")   # arm single-shot
            scope.trigger_burst()               # fire the burst
            voltages, sr, _ = scope.capture_triggered_waveform()
        """
        assert self._inst is not None, "Not connected."

        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, retries) + 1):
            done = self.wait_acquisition_done(timeout_s=timeout_s)
            if not done:
                warnings.warn(
                    f"Triggered acquisition did not complete within {timeout_s:.2f}s; "
                    "reading last available waveform record.",
                    RuntimeWarning,
                )

            for read_try in range(1, max(1, read_retries_per_capture) + 1):
                try:
                    voltages, y_inc, y_org = self.read_waveform(
                        channel=channel,
                        read_timeout_ms=read_timeout_ms,
                    )
                    if len(voltages) >= max(1, min_samples):
                        return voltages, y_inc, y_org
                    raise ValueError(
                        f"Waveform payload too short: {len(voltages)} samples "
                        f"(expected >= {max(1, min_samples)})."
                    )
                except Exception as exc:
                    last_error = exc
                    if read_try < max(1, read_retries_per_capture):
                        time.sleep(0.05)
                        continue

            if attempt < max(1, retries):
                self.clear_buffer()
                time.sleep(0.05)
                continue
            break

        raise RuntimeError(
            f"Failed to capture non-empty waveform after {max(1, retries)} attempt(s)."
        ) from last_error

    def read_waveform(
        self, channel: int = 1, read_timeout_ms: int = 5000
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
        # Parse IEEE 488.2 definite-length block exactly:
        #   #<n><n-digit-byte-count><payload>
        def _read_ieee_block_payload() -> bytes:
            old_read_term = self._inst.read_termination
            old_timeout = self._inst.timeout
            try:
                # IMPORTANT: disable termination for binary block reads.
                self._inst.read_termination = None
                self._inst.timeout = int(max(500, read_timeout_ms))

                # Select the waveform source and request one data block.
                self._inst.write(f":WAVeform:SOURce CHANnel{channel}")
                self._inst.write(":WAVeform:DATA:ALL?")

                # Read one raw USBTMC transfer and parse IEEE block from it.
                raw = self._inst.read_raw()
                if len(raw) < 3 or raw[0:1] != b"#":
                    raise ValueError(f"Unexpected waveform prefix: {raw[:32]!r}")

                n_digits = int(chr(raw[1]))
                if n_digits <= 0 or n_digits > 9:
                    raise ValueError(f"Invalid IEEE block digit count: {n_digits}")

                if len(raw) < 2 + n_digits:
                    raise ValueError("Incomplete IEEE header in raw transfer.")

                payload_len = int(raw[2 : 2 + n_digits].decode("ascii"))
                # Guard against corrupted header values that would stall reads.
                if payload_len < 0 or payload_len > 20_000_000:
                    raise ValueError(f"Unreasonable IEEE payload length: {payload_len}")

                header_len = 2 + n_digits
                payload_have = len(raw) - header_len

                # Firmware quirk: some DSO2D15 responses encode byte-count as
                # "header + payload" instead of payload only.
                if payload_have >= payload_len:
                    payload = raw[header_len : header_len + payload_len]
                elif payload_have == (payload_len - header_len):
                    payload = raw[header_len:]
                else:
                    raise ValueError(
                        f"Short IEEE payload read: got {payload_have} of {payload_len} bytes."
                    )

                return payload
            finally:
                self._inst.read_termination = old_read_term
                self._inst.timeout = old_timeout

        def _parse_packet(payload: bytes) -> Tuple[int, int, bytes]:
            """
            Parse one DATA:ALL payload block.

            Manual structure (after IEEE #9 header is removed):
              [0:9]   total waveform byte count
              [9:18]  uploaded bytes in this packet
              [18:]   waveform bytes (or metadata when uploaded == 0)
            """
            if len(payload) < 18 or (not payload[:18].isdigit()):
                raise ValueError(f"Unexpected packet header: {payload[:32]!r}")

            total_len = int(payload[0:9])
            uploaded_len = int(payload[9:18])

            if uploaded_len <= 0:
                # Metadata-only packet: keep polling for data packets.
                return total_len, 0, b""

            data_start = 18
            # On this firmware, uploaded_len does not always equal actual chunk bytes.
            # Use everything after the 18-byte ASCII header as waveform bytes.
            chunk = payload[data_start:]
            if not chunk:
                return total_len, uploaded_len, b""
            return total_len, uploaded_len, chunk

        # Read a few packets and keep the largest valid data chunk.
        # Empirically, this scope alternates metadata-only and data packets.
        best_chunk = b""
        expected_total = 0
        max_packet_reads = 6
        for _ in range(max_packet_reads):
            payload = _read_ieee_block_payload()
            total_len, _, chunk = _parse_packet(payload)
            if total_len > 0:
                expected_total = total_len
            if len(chunk) > len(best_chunk):
                best_chunk = chunk
            # Data packet is usually full enough once >= ~4000 bytes.
            if len(best_chunk) >= 3900:
                break

        if not best_chunk:
            raise ValueError("No waveform data chunk received.")

        binary_data = bytearray(best_chunk)
        if expected_total > 0 and len(binary_data) > expected_total:
            binary_data = binary_data[:expected_total]

        num_samples = len(binary_data)
        if num_samples == 0:
            raise ValueError("Empty waveform payload returned by instrument.")
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
        """
        Backward-compatible alias.

        Historically this method was (incorrectly) used for AUTO/NORMAL/SINGLE.
        Route those values to :TRIGger:SWEep, otherwise treat as trigger type.
        """
        mode_u = mode.upper()
        if mode_u in {"AUTO", "NORM", "NORMAL", "SING", "SINGLE"}:
            self.set_trigger_sweep(mode_u)
            return
        self._write(f":TRIGger:MODE {mode}")

    def set_trigger_type(self, trig_type: str) -> None:
        """Set trigger type (EDGE, PULSe, TV, SLOPe, TIMeout, WINdow, ...)."""
        self._write(f":TRIGger:MODE {trig_type}")

    def set_trigger_sweep(self, sweep: str) -> None:
        """Set trigger sweep mode (AUTO, NORMal, SINGle)."""
        sweep_map = {
            "AUTO": "AUTO",
            "NORM": "NORMal",
            "NORMAL": "NORMal",
            "SING": "SINGle",
            "SINGLE": "SINGle",
        }
        self._write(f":TRIGger:SWEep {sweep_map.get(sweep.upper(), sweep)}")

    def force_trigger(self) -> None:
        """Force one trigger event regardless of trigger condition."""
        self._write(":TRIGger:FORCe")
