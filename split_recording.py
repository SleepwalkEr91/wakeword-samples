#!/usr/bin/env python3
"""Split long phone recordings of a repeated wake phrase into single clips.

Record one file per person (voice memo, any format ffmpeg can read: m4a, mp3,
wav, ...) and say the wake phrase 20-50 times with ~2 seconds of silence in
between. This script cuts at the pauses and writes every repetition as a
16 kHz / mono / 16-bit WAV with a little padding - ready to add as real
recordings to an openWakeWord training set (e.g. EchoMuse's oww_forge).

Examples:
  python split_recording.py "recordings/raw/alice*" --name alice
  python split_recording.py take1.m4a take2.m4a --name bob
  python split_recording.py alice.m4a --name alice --thresh 12   # if detection is off

Clips go to <out>/<name>/. Pieces that are too short or too long go to
<out>/<name>/_check/ for listening (usually a cough, a laugh, or two
repetitions stuck together). Requires ffmpeg on PATH.
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
except ImportError:
    sys.exit("pydub is missing: pip install pydub")


def main():
    p = argparse.ArgumentParser(
        description="Split recordings at pauses into single wake word clips",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("files", nargs="+", help="recording file(s); wildcards allowed (quote them)")
    p.add_argument("--name", required=True, help="speaker tag, e.g. alice")
    p.add_argument("--out", default="recordings", help="output directory (default: recordings)")
    p.add_argument("--min-silence", type=int, default=600,
                   help="pause length in ms that separates clips (default 600)")
    p.add_argument("--thresh", type=float, default=16,
                   help="silence = this many dB below the average level (default 16; "
                        "lower = more sensitive)")
    p.add_argument("--pad", type=int, default=200,
                   help="padding in ms before/after each clip (default 200)")
    p.add_argument("--min-len", type=float, default=0.5,
                   help="shorter pieces go to _check (seconds, default 0.5)")
    p.add_argument("--max-len", type=float, default=2.5,
                   help="longer pieces go to _check (seconds, default 2.5)")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")

    # PowerShell does not expand wildcards for native programs, so do it here.
    files = []
    for f in args.files:
        hits = sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
        if not hits:
            print(f"no files match {f}")
        files.extend(hits)
    if not files:
        sys.exit("no recordings found")

    out = Path(args.out) / args.name
    check = out / "_check"
    out.mkdir(parents=True, exist_ok=True)

    total_ok = total_check = 0
    for f in files:
        src = Path(f)
        audio = AudioSegment.from_file(src).set_channels(1).set_frame_rate(16000).set_sample_width(2)
        thresh = audio.dBFS - args.thresh
        ranges = detect_nonsilent(audio, min_silence_len=args.min_silence,
                                  silence_thresh=thresh, seek_step=10)
        print(f"{src.name}: {len(audio) / 1000:.0f}s, {len(ranges)} pieces found "
              f"(silence below {thresh:.0f} dBFS)")
        ok = bad = 0
        for i, (start, end) in enumerate(ranges):
            s, e = max(0, start - args.pad), min(len(audio), end + args.pad)
            clip = audio[s:e]
            length = (end - start) / 1000
            good = args.min_len <= length <= args.max_len
            target = out if good else check
            target.mkdir(parents=True, exist_ok=True)
            name = f"rec_{args.name}_{src.stem}_{i:03d}.wav"
            clip.export(target / name, format="wav")
            if good:
                ok += 1
            else:
                bad += 1
                print(f"  -> _check: {name} ({length:.2f}s)")
        print(f"  {ok} clips ok, {bad} to check")
        total_ok += ok
        total_check += bad

    print(f"\nDone: {total_ok} clips in {out}" +
          (f", {total_check} in {check} to listen to" if total_check else ""))
    print("Tip: spot-check a few; move good pieces from _check into the main folder.")


if __name__ == "__main__":
    main()
