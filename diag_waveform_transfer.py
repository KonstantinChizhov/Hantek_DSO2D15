import time
from typing import Optional

from dso2d15 import DSO2D15, Coupling


CHANNEL = 1
ITERATIONS = 8
MAX_PACKETS_PER_CAPTURE = 20
READ_TIMEOUT_MS = 4000


def _wait_triggered(scope: DSO2D15, timeout_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = scope._query(":TRIGger:STATus?").strip().upper()
            if "TRIG" in status and "NOTRIG" not in status:
                return True
        except Exception:
            pass
        time.sleep(0.03)
    return False


def _read_one_block(scope: DSO2D15) -> dict:
    inst = scope._inst
    assert inst is not None

    old_term = inst.read_termination
    old_timeout = inst.timeout
    try:
        inst.read_termination = None
        inst.timeout = READ_TIMEOUT_MS
        inst.write(f":WAVeform:SOURce CHANnel{CHANNEL}")
        inst.write(":WAVeform:DATA:ALL?")
        raw = inst.read_raw()
    finally:
        inst.read_termination = old_term
        inst.timeout = old_timeout

    out = {
        "raw_len": len(raw),
        "ok": False,
        "n_digits": None,
        "payload_len": None,
        "payload_have": None,
        "payload": b"",
        "err": "",
    }

    if len(raw) < 3 or raw[:1] != b"#":
        out["err"] = f"bad-prefix {raw[:24]!r}"
        return out

    try:
        n_digits = int(chr(raw[1]))
        out["n_digits"] = n_digits
        if n_digits < 1 or n_digits > 9:
            out["err"] = f"bad-n-digits {n_digits}"
            return out

        if len(raw) < 2 + n_digits:
            out["err"] = "short-header"
            return out

        payload_len = int(raw[2 : 2 + n_digits].decode("ascii"))
        out["payload_len"] = payload_len
        payload_have = len(raw) - (2 + n_digits)
        out["payload_have"] = payload_have
        out["payload"] = raw[2 + n_digits : 2 + n_digits + max(payload_len, 0)]
        out["ok"] = True
        return out
    except Exception as exc:
        out["err"] = f"parse-failed {type(exc).__name__}: {exc}"
        return out


def _parse_packet_payload(payload: bytes) -> tuple[Optional[int], Optional[int], int, bool]:
    if len(payload) < 18 or (not payload[:18].isdigit()):
        return None, None, 0, False
    total_len = int(payload[0:9])
    uploaded_len = int(payload[9:18])
    if uploaded_len <= 0:
        return total_len, uploaded_len, 0, True
    data_start = 18
    chunk_len = max(0, min(uploaded_len, len(payload) - data_start))
    return total_len, uploaded_len, chunk_len, True


def main() -> None:
    scope = DSO2D15.auto_discover()
    assert scope._inst is not None

    try:
        print("=== Diagnostic: waveform transfer ===")
        scope.enable_channel(CHANNEL)
        scope.set_channel_coupling(CHANNEL, Coupling.DC)
        scope.set_channel_scale(CHANNEL, 0.5)
        scope.set_timebase_scale(0.0005)
        scope.set_acquisition_points(4000)
        scope.set_trigger_edge_source(CHANNEL)
        scope.set_trigger_level(0.0)
        scope.set_trigger_sweep("SINGle")

        for i in range(1, ITERATIONS + 1):
            print(f"\n--- Capture {i}/{ITERATIONS} ---")
            try:
                scope.clear_buffer()
            except Exception as exc:
                print(f"clear_buffer warning: {type(exc).__name__}: {exc}")

            scope.set_trigger_sweep("SINGle")
            time.sleep(0.03)
            scope.force_trigger()
            trig_ok = _wait_triggered(scope, timeout_s=2.0)
            print(f"triggered={trig_ok}")

            collected = 0
            expected_total = None
            for pkt in range(1, MAX_PACKETS_PER_CAPTURE + 1):
                try:
                    blk = _read_one_block(scope)
                except Exception as exc:
                    print(f"pkt{pkt:02d} read-exc: {type(exc).__name__}: {exc}")
                    break

                if not blk["ok"]:
                    print(f"pkt{pkt:02d} invalid: {blk['err']}; raw_len={blk['raw_len']}")
                    break

                total_len, uploaded_len, chunk_len, header_ok = _parse_packet_payload(
                    blk["payload"]
                )
                if not header_ok:
                    pfx = blk["payload"][:24]
                    print(
                        f"pkt{pkt:02d} header-bad raw_len={blk['raw_len']} "
                        f"p_len={blk['payload_len']} p_have={blk['payload_have']} prefix={pfx!r}"
                    )
                    break

                if expected_total is None and total_len is not None and total_len > 0:
                    expected_total = total_len

                collected += chunk_len
                print(
                    f"pkt{pkt:02d} raw={blk['raw_len']} p_len={blk['payload_len']} "
                    f"p_have={blk['payload_have']} total={total_len} uploaded={uploaded_len} "
                    f"chunk={chunk_len} collected={collected}"
                )

                if expected_total is not None and collected >= expected_total:
                    print(f"complete: collected={collected} expected={expected_total}")
                    break

            if expected_total is not None and collected < expected_total:
                print(f"incomplete: collected={collected} expected={expected_total}")

    finally:
        scope.close()


if __name__ == "__main__":
    main()
