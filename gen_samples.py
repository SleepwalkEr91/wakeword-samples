#!/usr/bin/env python3
"""Generate positive training samples for a custom openWakeWord wake word.

Built for EchoMuse's oww_forge, but the output is plain 16 kHz mono 16-bit WAV
and works with any openWakeWord training pipeline.

Why: the standard pipeline synthesizes positives with American English voices
only. If your household says the wake phrase with a different accent or in a
different language, those clips teach the model the wrong pronunciation. This
script produces clips from native speakers of the language you choose.

Providers:
  edge    Microsoft Edge TTS (free; voices of the chosen locale, optionally
          all *Multilingual* voices)
  openai  OpenAI gpt-4o-mini-tts (pronunciation steerable via instructions)
  eleven  ElevenLabs (multilingual model with an explicit language code;
          Voice Library voices need a paid plan for API use)

Output: <out>/<provider>/<provider>_<voice>_<n>.wav, silence trimmed, ~150 ms
padding. Existing files are skipped, so any run can be resumed or topped up
by raising --per-voice.

Examples:
  python gen_samples.py --providers eleven --test --phrase "Hey Jarvis"
  python gen_samples.py --providers eleven --eleven-library-only --per-voice 60
  python gen_samples.py --providers edge --voices Florian,Seraphina --per-voice 40
  python gen_samples.py --providers openai --lang fr --phrase "Salut Jarvis"

API keys come from the environment: OPENAI_API_KEY, ELEVENLABS_API_KEY.
Requires ffmpeg on PATH.
"""
import argparse
import asyncio
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

DEFAULT_PHRASE = "Hey Jarvis"

# Edge TTS: variation through rate and pitch
EDGE_RATES = ["-20%", "-10%", "-5%", "+0%", "+5%", "+10%", "+20%"]
EDGE_PITCHES = ["-12Hz", "-6Hz", "-3Hz", "+0Hz", "+3Hz", "+6Hz", "+12Hz"]

# OpenAI: all current voices; unsupported ones are skipped automatically
OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova",
                 "onyx", "sage", "shimmer", "verse", "marin", "cedar"]
# gpt-4o-mini-tts has no speed parameter -> vary the speaking style instead
OPENAI_STYLES = [
    "Speak calmly at a normal pace.",
    "Speak a little faster and casually, as in everyday life.",
    "Speak slowly and clearly.",
    "Speak quietly and a little tired.",
    "Speak in a friendly, slightly questioning tone.",
    "Speak louder, as if calling from the next room.",
    "Speak briefly and firmly.",
]
DEFAULT_HINT = ("You are a native speaker of the language with code '{lang}' and speak it "
                "with a natural native accent. Say exactly the given text and nothing else.")

ELEVEN_LANG_MODELS = ("eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3")

# Remove leading/trailing silence, then add 150 ms padding
TRIM_FILTER = (
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
    "areverse,"
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
    "areverse,"
    "adelay=150:all=1,apad=pad_dur=0.15"
)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")


def default_phrases(base: str):
    """A few punctuation variants of one phrase -> different intonation."""
    return [base, f"{base}!", base.replace(" ", ", ", 1) + ".", f"{base}?"]


def matches(name: str, patterns) -> bool:
    n = name.lower()
    return any(p in n for p in patterns)


def filter_voices(voices, args):
    """voices: list of (display_name, id). Filter by --voices / --exclude (substring)."""
    if args.voices:
        want = [v.strip().lower() for v in args.voices.split(",") if v.strip()]
        voices = [v for v in voices if matches(v[0], want) or matches(v[1], want)]
    if args.exclude:
        drop = [v.strip().lower() for v in args.exclude.split(",") if v.strip()]
        voices = [v for v in voices if not (matches(v[0], drop) or matches(v[1], drop))]
    return voices


def finalize(src: Path, dst: Path, max_s: float) -> str:
    """Convert to 16 kHz mono s16 and trim silence. Returns '' on success, else a reason."""
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-af", TRIM_FILTER, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "ffmpeg: " + res.stderr.strip()[:200]
    with wave.open(str(tmp)) as w:
        dur = w.getnframes() / w.getframerate()
    if dur < 0.5 or dur > max_s:
        tmp.unlink(missing_ok=True)
        return f"length {dur:.2f}s outside 0.5-{max_s}s (probably misread)"
    tmp.replace(dst)
    return ""


def plan(provider, voices, args, out_dir):
    """Build jobs (voice_name, voice_id, index, rng, dest), skipping existing files."""
    jobs = []
    n = 1 if args.test else args.per_voice
    for name, vid in voices:
        for i in range(n):
            dest = out_dir / f"{provider}_{safe(name)}_{i:03d}.wav"
            if dest.exists():
                continue
            jobs.append((name, vid, i, random.Random(f"{provider}|{vid}|{i}"), dest))
    return jobs


def pick_phrase(rng, args):
    return args.phrases[0] if args.test else rng.choice(args.phrases)


def api_error(e) -> str:
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    return f"HTTP {status}: {body}" if status else str(e)[:300]


class VoiceGuard:
    """Skip a voice after 3 consecutive failures."""
    def __init__(self):
        self.fails = {}

    def dead(self, vid):
        return self.fails.get(vid, 0) >= 3

    def ok(self, vid):
        self.fails[vid] = 0

    def fail(self, vid, name, err):
        self.fails[vid] = self.fails.get(vid, 0) + 1
        print(f"  ! {name}: {err}")
        if self.fails[vid] == 3:
            print(f"  ! {name}: 3 failures in a row - skipping this voice")


# ---------------------------------------------------------------- Edge TTS
def run_edge(args, out_dir, tmp_dir):
    try:
        import edge_tts
    except ImportError:
        sys.exit("edge-tts is missing: pip install edge-tts")

    async def main():
        all_voices = await edge_tts.list_voices()
        prefix = args.lang.lower() + "-"
        voices = [(v["ShortName"], v["ShortName"]) for v in all_voices
                  if v["Locale"].lower().startswith(prefix)
                  or (args.edge_multilingual and "Multilingual" in v["ShortName"])]
        voices = filter_voices(voices, args)
        print(f"[edge] {len(voices)} voices: {', '.join(v[0] for v in voices)}")
        jobs = plan("edge", voices, args, out_dir)
        guard, done = VoiceGuard(), 0
        for k, (name, vid, i, rng, dest) in enumerate(jobs, 1):
            if guard.dead(vid):
                continue
            text = pick_phrase(rng, args)
            rate = "+0%" if args.test else rng.choice(EDGE_RATES)
            pitch = "+0Hz" if args.test else rng.choice(EDGE_PITCHES)
            raw = tmp_dir / (dest.stem + ".mp3")
            try:
                await edge_tts.Communicate(text, vid, rate=rate, pitch=pitch).save(str(raw))
                err = finalize(raw, dest, args.max_seconds)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
            if err:
                guard.fail(vid, name, err)
            else:
                guard.ok(vid)
                done += 1
            if k % 25 == 0:
                print(f"[edge] {k}/{len(jobs)}")
        return done

    return asyncio.run(main())


# ---------------------------------------------------------------- OpenAI
def run_openai(args, out_dir, tmp_dir):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    from openai import OpenAI
    client = OpenAI()

    hint = args.hint or DEFAULT_HINT.format(lang=args.lang)
    voices = filter_voices([(v, v) for v in OPENAI_VOICES], args)
    print(f"[openai] {len(voices)} voices, model {args.openai_model}")
    jobs = plan("openai", voices, args, out_dir)
    guard, done = VoiceGuard(), 0
    for k, (name, vid, i, rng, dest) in enumerate(jobs, 1):
        if guard.dead(vid):
            continue
        text = pick_phrase(rng, args)
        style = OPENAI_STYLES[0] if args.test else rng.choice(OPENAI_STYLES)
        raw = tmp_dir / (dest.stem + ".raw.wav")
        try:
            resp = client.audio.speech.create(
                model=args.openai_model, voice=vid, input=text,
                instructions=f"{hint} {style}", response_format="wav")
            raw.write_bytes(resp.content)
            err = finalize(raw, dest, args.max_seconds)
        except Exception as e:  # noqa: BLE001
            err = api_error(e)
        if err:
            guard.fail(vid, name, err)
        else:
            guard.ok(vid)
            done += 1
        if k % 25 == 0:
            print(f"[openai] {k}/{len(jobs)}")
    return done


# ---------------------------------------------------------------- ElevenLabs
def run_eleven(args, out_dir, tmp_dir):
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("ELEVENLABS_API_KEY is not set")
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=key)

    all_voices = []
    try:
        token = None
        while True:
            kw = {"page_size": 100}
            if token:
                kw["next_page_token"] = token
            resp = client.voices.search(**kw)
            all_voices.extend(resp.voices)
            token = getattr(resp, "next_page_token", None)
            if not getattr(resp, "has_more", False) or not token:
                break
    except (AttributeError, TypeError):
        all_voices = list(client.voices.get_all().voices)
    if args.eleven_library_only:
        all_voices = [v for v in all_voices if getattr(v, "category", None) != "premade"]
    voices = filter_voices([(v.name, v.voice_id) for v in all_voices], args)
    print(f"[eleven] {len(voices)} voices, model {args.eleven_model}: "
          f"{', '.join(v[0] for v in voices)}")
    use_lang = args.eleven_model in ELEVEN_LANG_MODELS
    if not use_lang:
        print(f"[eleven] note: {args.eleven_model} takes no language code - "
              f"the language is guessed from the text")

    jobs = plan("eleven", voices, args, out_dir)
    guard, done = VoiceGuard(), 0
    for k, (name, vid, i, rng, dest) in enumerate(jobs, 1):
        if guard.dead(vid):
            continue
        text = pick_phrase(rng, args)
        settings = VoiceSettings(
            stability=0.5 if args.test else round(rng.uniform(0.3, 0.75), 2),
            similarity_boost=0.75, style=0.0,
            speed=1.0 if args.test else round(rng.uniform(0.85, 1.15), 2))
        kwargs = dict(voice_id=vid, text=text, model_id=args.eleven_model,
                      output_format="pcm_16000", voice_settings=settings)
        if use_lang:
            kwargs["language_code"] = args.lang
        raw = tmp_dir / (dest.stem + ".raw.wav")
        try:
            pcm = b"".join(client.text_to_speech.convert(**kwargs))
            with wave.open(str(raw), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm)
            err = finalize(raw, dest, args.max_seconds)
        except Exception as e:  # noqa: BLE001
            err = api_error(e)
        if err:
            guard.fail(vid, name, err)
        else:
            guard.ok(vid)
            done += 1
        if k % 25 == 0:
            print(f"[eleven] {k}/{len(jobs)}")
    return done


RUNNERS = {"edge": run_edge, "openai": run_openai, "eleven": run_eleven}


def main():
    p = argparse.ArgumentParser(
        description="Generate wake word training samples in a chosen language",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--providers", default="eleven",
                   help="comma-separated: edge,openai,eleven (default: eleven)")
    p.add_argument("--lang", default="de",
                   help="language code, e.g. de, fr, en (default: de). Edge: locale prefix; "
                        "ElevenLabs: language_code; OpenAI: used in the default instruction")
    p.add_argument("--phrase", action="append", dest="phrases",
                   help=f"phrase to speak (repeatable). Default: punctuation variants of "
                        f"'{DEFAULT_PHRASE}'. A single --phrase is also expanded into variants")
    p.add_argument("--per-voice", type=int, default=30, help="clips per voice (default 30)")
    p.add_argument("--test", action="store_true",
                   help="only 1 neutral clip per voice into <out>/test for listening")
    p.add_argument("--out", default="samples", help="output directory (default: samples)")
    p.add_argument("--voices", help="only these voices (comma-separated substrings of name/id)")
    p.add_argument("--exclude", help="skip these voices (comma-separated substrings)")
    p.add_argument("--edge-multilingual", action="store_true",
                   help="Edge: also include every *Multilingual* voice of other locales")
    p.add_argument("--eleven-library-only", action="store_true",
                   help="ElevenLabs: only your own/Voice Library voices, not the premade ones")
    p.add_argument("--hint", help="pronunciation instruction for OpenAI (overrides the default)")
    p.add_argument("--openai-model", default="gpt-4o-mini-tts")
    p.add_argument("--eleven-model", default="eleven_flash_v2_5")
    p.add_argument("--max-seconds", type=float, default=2.5,
                   help="longer clips are discarded (default 2.5)")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    if not args.phrases:
        args.phrases = default_phrases(DEFAULT_PHRASE)
    elif len(args.phrases) == 1:
        args.phrases = default_phrases(args.phrases[0])

    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    unknown = [x for x in providers if x not in RUNNERS]
    if unknown:
        sys.exit(f"unknown provider(s): {', '.join(unknown)}")

    base = Path(args.out) / ("test" if args.test else "")
    total = {}
    with tempfile.TemporaryDirectory() as td:
        for prov in providers:
            out_dir = base / prov
            out_dir.mkdir(parents=True, exist_ok=True)
            total[prov] = RUNNERS[prov](args, out_dir, Path(td))

    print("\nDone:")
    for prov, n in total.items():
        have = len(list((base / prov).glob("*.wav")))
        print(f"  {prov}: {n} new, {have} total in {base / prov}")


if __name__ == "__main__":
    main()
