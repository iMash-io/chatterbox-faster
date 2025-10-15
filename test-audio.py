#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_sse_play.py — OpenAI-compatible TTS SSE client with true realtime playback

- Parses SSE (response.metadata + response.output_audio.delta)
- Decodes base64 PCM / f32 / (mu-/a-law passthrough to ffplay)
- Plays in realtime via sounddevice (CoreAudio) or ffplay
- Prints TTFB on first audio bytes
- Optional: write WAV (--wav) and raw dump (--save-raw)

Mac fix: sounddevice backend now uses a safe RawOutputStream callback with a thread-safe byte queue.
"""

import argparse
import base64
import io
import json
import queue
import signal
import subprocess
import sys
import time
import wave
from typing import Generator, Optional, Tuple

import requests


# ---------- SSE utilities ----------

def sse_events(resp: requests.Response):
    """Yield (event_name_or_None, data_str) from a text/event-stream response."""
    event = None
    data_lines = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\n")
        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    if data_lines:
        yield event, "\n".join(data_lines)


def detect_sample_width(fmt: str) -> int:
    f = (fmt or "").lower()
    if f in ("pcm16", "s16le"): return 2
    if f in ("pcm_f32", "f32le"): return 4
    if f in ("mulaw", "alaw"): return 1
    return 2


# ---------- ffplay backend ----------

def start_ffplay(sample_rate: int, fmt: str, loglevel: str) -> Optional[subprocess.Popen]:
    """Start ffplay reading raw audio from stdin (no -ac to avoid macOS flag issues)."""
    fmt_map = {
        "pcm16": "s16le",
        "s16le": "s16le",
        "pcm_f32": "f32le",
        "f32le": "f32le",
        "mulaw": "mulaw",
        "alaw": "alaw",
    }
    ff_fmt = fmt_map.get((fmt or "").lower(), "s16le")
    cmd = [
        "ffplay",
        "-hide_banner",
        "-loglevel", loglevel,
        "-nodisp",
        "-autoexit",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", ff_fmt,
        "-ar", str(sample_rate),
        "-i", "pipe:0",
    ]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except FileNotFoundError:
        print("❌ ffplay not found. Install FFmpeg or use --backend sd.")
        return None


# ---------- sounddevice backend (fixed) ----------

class SDPlayer:
    """
    Realtime audio player using sounddevice.RawOutputStream + a bounded byte queue.

    - Supports pcm16 (int16) and f32le (float32).
    - Low-latency: bounded queue, drops old data if producer outruns consumer.
    """
    def __init__(self, samplerate: int, channels: int, fmt: str,
                 max_queue_secs: float = 0.5):
        import sounddevice as sd  # local import so script works without it if using ffplay
        self.sd = sd
        self.samplerate = int(samplerate)
        self.channels = max(1, int(channels))
        self.dtype = self._dtype_for(fmt)
        self.sample_bytes = detect_sample_width(fmt)
        self.frame_bytes = self.sample_bytes * self.channels
        # ~20ms packet -> frames_per_packet ~ samplerate * 0.02
        approx_packet_frames = max(1, int(self.samplerate * 0.02))
        max_frames = int(max_queue_secs * self.samplerate)
        max_packets = max(2, max_frames // approx_packet_frames)
        self.q: queue.Queue[bytes] = queue.Queue(maxsize=max_packets)
        self.closed = False

        def _callback(outdata, frames, time_info, status):
            if status:
                # underrun/overrun messages
                print(status, file=sys.stderr)
            needed = frames * self.frame_bytes
            got = 0
            # Fill a local buffer from queued audio
            # (We avoid numpy entirely to keep this portable.)
            buf = bytearray(needed)
            mv = memoryview(buf)
            # Pull queued packets until we fill or queue empties
            while got < needed and not self.closed:
                try:
                    chunk = self.q.get_nowait()
                except queue.Empty:
                    break
                take = min(len(chunk), needed - got)
                mv[got:got+take] = chunk[:take]
                got += take
                # If chunk longer than we needed, push remainder back to front
                if take < len(chunk):
                    # put remainder at the front by rebuilding queue (rare path)
                    remainder = chunk[take:]
                    # It's OK to drop; but we try to preserve by immediate put_nowait
                    try:
                        self.q.put_nowait(remainder)
                    except queue.Full:
                        pass
                    break
            # If not enough data, remaining bytes stay zero (silence).
            outdata[:needed] = buf

        self.stream = self.sd.RawOutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=_callback,
            blocksize=0,          # backend decides; latency governed by feed rate & queue depth
        )
        self.stream.start()

    @staticmethod
    def _dtype_for(fmt: str) -> str:
        f = (fmt or "").lower()
        if f in ("pcm_f32", "f32le"):
            return "float32"
        return "int16"

    def write(self, raw_bytes: bytes):
        if self.closed:
            return
        try:
            self.q.put_nowait(raw_bytes)
        except queue.Full:
            # Drop old audio to keep latency bounded.
            with self.q.mutex:
                self.q.queue.clear()
            try:
                self.q.put_nowait(raw_bytes)
            except queue.Full:
                pass

    def close(self):
        self.closed = True
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass


# ---------- WAV writer ----------

def open_wav(path: str, channels: int, samplerate: int, fmt: str) -> Optional[wave.Wave_write]:
    if (fmt or "").lower() not in ("pcm16", "s16le", "pcm_f32", "f32le"):
        print("ℹ️  Skipping WAV write (non-PCM stream).")
        return None
    wf = wave.open(path, "wb")
    sampwidth = 2 if (fmt or "").lower() in ("pcm16", "s16le") else 4
    wf.setnchannels(max(1, channels))
    wf.setsampwidth(sampwidth)
    wf.setframerate(samplerate)
    return wf


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Realtime OpenAI-compatible TTS SSE client.")
    parser.add_argument("--url", default="https://kamj9wzldv1f7v-8000.proxy.runpod.net/v1/audio/speech")
    parser.add_argument("--model", default="faster_multi_api")
    parser.add_argument("--text", default="This is an example of what I want the TTS today.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--play", action="store_true", help="Enable audio playback.")
    parser.add_argument("--backend", choices=["sd", "ffplay"], default="sd", help="Audio backend (sd = sounddevice).")
    parser.add_argument("--ffplay-loglevel", default="warning",
                        choices=["quiet","panic","fatal","error","warning","info","verbose","debug","trace"])
    parser.add_argument("--wav", default=None, help="Save a WAV while streaming (PCM only).")
    parser.add_argument("--save-raw", default=None, help="Save raw concatenated bytes.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--no-stream-flag", action="store_true",
                        help="Omit 'stream': true (tests non-SSE fallback).")
    args = parser.parse_args()

    payload = {"model": args.model, "input": args.text, "language": args.language}
    if not args.no_stream_flag:
        payload["stream"] = True
    if args.voice:
        payload["voice"] = args.voice

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    interrupted = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__("flag", True))

    print("Sending request...")
    t0 = time.time()
    try:
        resp = requests.post(args.url, headers=headers, json=payload,
                             stream=True, timeout=args.timeout, verify=not args.insecure)
    except Exception as e:
        print(f"❌ HTTP error: {e}")
        sys.exit(1)

    ctype = resp.headers.get("Content-Type", "")

    # Defaults (overwritten by metadata)
    samplerate = 24000
    fmt = "pcm16"
    channels = 1

    # outputs
    wav_writer: Optional[wave.Wave_write] = None
    raw_sink: Optional[io.BufferedWriter] = None
    sd_player: Optional[SDPlayer] = None
    ff_proc: Optional[subprocess.Popen] = None

    def cleanup():
        try:
            if sd_player: sd_player.close()
        except Exception:
            pass
        try:
            if ff_proc and ff_proc.stdin:
                try: ff_proc.stdin.close()
                except Exception: pass
            if ff_proc:
                try: ff_proc.wait(timeout=3)
                except Exception: pass
        except Exception:
            pass
        try:
            if wav_writer: wav_writer.close()
        except Exception:
            pass
        try:
            if raw_sink: raw_sink.close()
        except Exception:
            pass

    # Non-SSE fallback (e.g., full WAV body)
    if "text/event-stream" not in ctype:
        body = resp.content or b""
        ttfb_ms = (time.time() - t0) * 1000.0
        print(f"⚠️  Non-SSE response ({ctype}). TTFB={ttfb_ms:.2f} ms, {len(body)} bytes.")
        if args.wav:
            with open(args.wav, "wb") as f: f.write(body)
            print(f"💾 Saved body to {args.wav}")
        if args.play and args.backend == "ffplay":
            ff_proc = start_ffplay(samplerate, fmt, args.ffplay_loglevel)
            if ff_proc and ff_proc.stdin:
                ff_proc.stdin.write(body)
                ff_proc.stdin.close()
                ff_proc.wait()
        elif args.play and args.backend == "sd":
            try:
                import sounddevice as _  # noqa: F401
            except Exception:
                print("❌ sounddevice not installed. `pip install sounddevice` or use --backend ffplay.")
                cleanup(); sys.exit(1)
            try:
                sd_player = SDPlayer(samplerate, 1, "pcm16")
                sd_player.write(body)
                # crude drain:
                time.sleep(len(body) / (2 * samplerate))
            finally:
                cleanup()
        sys.exit(0)

    # SSE streaming
    first_delta = True
    total_bytes = 0
    try:
        for event, data_str in sse_events(resp):
            if interrupted["flag"]:
                print("⏹️  Interrupted by user.")
                break

            if event == "response.metadata":
                try:
                    meta = json.loads(data_str)
                except Exception as e:
                    print(f"Metadata parse error: {e}", file=sys.stderr)
                    continue
                samplerate = int(meta.get("sample_rate", samplerate))
                fmt = str(meta.get("format", fmt))
                channels = int(meta.get("channels", channels))
                print(f"🔎 Metadata: sample_rate={samplerate}, format={fmt}, channels={channels}")

                # prepare outputs
                if args.wav and wav_writer is None:
                    wav_writer = open_wav(args.wav, channels, samplerate, fmt)

                if args.save_raw and raw_sink is None:
                    raw_sink = open(args.save_raw, "wb")

                if args.play and sd_player is None and args.backend == "sd":
                    try:
                        sd_player = SDPlayer(samplerate, channels, fmt)
                        print("🔊 sounddevice stream started.")
                    except Exception as e:
                        print(f"❌ sounddevice init failed: {e}")
                        print("ℹ️  Falling back to ffplay.")
                        args.backend = "ffplay"

                if args.play and args.backend == "ffplay" and ff_proc is None:
                    ff_proc = start_ffplay(samplerate, fmt, args.ffplay_loglevel)
                    if ff_proc:
                        print("🔊 ffplay started.")

            elif event == "response.output_audio.delta":
                try:
                    delta = json.loads(data_str)
                    b64 = delta.get("audio", "")
                    if not b64:
                        continue
                    chunk = base64.b64decode(b64)
                except Exception as e:
                    print(f"Delta parse/decode error: {e}", file=sys.stderr)
                    continue

                if first_delta:
                    ttfb_ms = (time.time() - t0) * 1000.0
                    print(f"⏱️  Time to first audio byte: {ttfb_ms:.2f} ms")
                    print(f"📦 First PCM bytes (len={len(chunk)}): {chunk[:25]!r}")
                    first_delta = False

                total_bytes += len(chunk)

                # play
                if args.play:
                    if args.backend == "sd" and sd_player:
                        sd_player.write(chunk)
                    elif args.backend == "ffplay" and ff_proc and ff_proc.stdin:
                        try:
                            ff_proc.stdin.write(chunk)
                            ff_proc.stdin.flush()
                        except BrokenPipeError:
                            print("⚠️  ffplay closed stdin. Stopping playback.")
                            ff_proc = None

                # save
                if wav_writer:
                    try: wav_writer.writeframes(chunk)
                    except Exception as e:
                        print(f"⚠️  WAV write error: {e}")
                        wav_writer.close(); wav_writer = None
                if raw_sink:
                    try: raw_sink.write(chunk)
                    except Exception as e:
                        print(f"⚠️  Raw save error: {e}")
                        raw_sink.close(); raw_sink = None

            elif event == "response.completed":
                print("✅ Stream completed.")
                break
            # ignore other events

    finally:
        cleanup()

    # stats (approx for PCM)
    bps = detect_sample_width(fmt) * max(1, channels) * float(samplerate)
    seconds = (total_bytes / bps) if bps else 0.0
    print(f"🎧 Received ~{seconds:.2f} s audio ({total_bytes} bytes @ {samplerate} Hz, {channels} ch, {fmt}).")


if __name__ == "__main__":
    main()

