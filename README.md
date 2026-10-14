# wakeword-samples

Generate **native-language positive samples** for custom
[openWakeWord](https://github.com/dscripka/openWakeWord) models, and turn your
own phone recordings into training clips.

Built alongside EchoMuse's
[`oww_forge`](https://github.com/wilbowes/EchoMuse/tree/main/oww_forge), but the
output is plain 16 kHz / mono / 16-bit WAV and works with any openWakeWord
training pipeline.

## Why

The standard openWakeWord pipeline synthesizes its positive clips with
**American English** voices only. If your household says the wake phrase with a
different accent or in a different language (say, a German "Hey Jarvis"), the
model mostly learns a pronunciation nobody at home uses. Mixing in clips from
native speakers, plus a few real recordings, pulls the model toward the voices
that matter.

## What's in here

| Script | Purpose |
|---|---|
| `gen_samples.py` | Synthesizes the wake phrase with native voices from ElevenLabs, OpenAI or Microsoft Edge TTS |
| `split_recording.py` | Cuts long phone recordings ("Hey Jarvis … Hey Jarvis …") into one clip per repetition |

## Setup

Both scripts are plain Python and run natively on **Windows, Linux and macOS** –
no WSL or Docker needed for them. They require **Python 3.10–3.12** and
**ffmpeg** on `PATH`.

### Windows (PowerShell)

```powershell
winget install Python.Python.3.12
winget install ffmpeg            # open a NEW terminal afterwards so PATH is updated

cd C:\path\to\wakeword-samples
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1     # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

$env:ELEVENLABS_API_KEY = "..."  # valid for this terminal only
$env:OPENAI_API_KEY = "..."      # only for --providers openai
```

Inside the activated venv, always run `python …`, not `py …` – `py` may pick a
different Python installation without the packages.

### Linux / macOS

```bash
sudo apt install python3 python3-venv ffmpeg     # macOS: brew install python ffmpeg

cd /path/to/wakeword-samples
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ELEVENLABS_API_KEY=...
export OPENAI_API_KEY=...        # only for --providers openai
```

The examples below use `python` and forward slashes; both work in PowerShell
and in bash.

## Generating samples

### 1. Listen first

Generate one neutral clip per voice and listen to how each one says the phrase:

```bash
python gen_samples.py --providers eleven --test --phrase "Hey Jarvis"
```

Clips land in `samples/test/<provider>/`, with the voice name in the file name.
Note which voices get the pronunciation right.

### 2. Generate the real set

```bash
python gen_samples.py --providers eleven --voices Daniel,Helmut,Lea --per-voice 60
```

Output: `samples/<provider>/<provider>_<voice>_<n>.wav`, silence trimmed with
~150 ms padding. Existing files are skipped, so a run can be resumed at any time,
and raising `--per-voice` later simply tops the set up.

Each clip varies the phrase's punctuation (and therefore its intonation) plus
provider-specific parameters: speed and stability (ElevenLabs), rate and pitch
(Edge), speaking style (OpenAI).

### Providers

| Provider | Cost | Notes |
|---|---|---|
| `eleven` | ~2 credits per clip with `eleven_flash_v2_5` (measured on a short two-word phrase) | Best quality by far. Sends an explicit language code. **Voice Library voices need a paid plan for API use** (`paid_plan_required`); add native speakers of your language to *My Voices*, then use `--eleven-library-only` |
| `openai` | cents per few hundred clips | `gpt-4o-mini-tts`; pronunciation is steered through an instruction (`--hint`) |
| `edge` | free | Voices of the `--lang` locale; `--edge-multilingual` adds the *Multilingual* voices of other locales. Quality varies a lot between voices |

**What has actually been tested:** ElevenLabs and Edge, for a German
"Hey Jarvis", on **Windows only** (scripts in PowerShell, oww_forge in Docker
under WSL2). Linux and macOS should work the same way but have not been tried.
ElevenLabs with native German voices from the Voice Library was
far better – natural, varied, and every voice got the pronunciation right.
Edge worked for only a handful of voices; most pure German voices read the name
with German spelling rules ("Iarvis"). The OpenAI provider is implemented but
**untested**; it should work, and its quality is likely somewhere between the two.

### Useful options

| Option | Meaning |
|---|---|
| `--lang de` | Target language (default `de`) |
| `--phrase "…"` | Phrase to speak; repeatable. A single phrase is expanded into punctuation variants |
| `--voices a,b` / `--exclude a,b` | Filter voices by substring of name or id |
| `--per-voice N` | Clips per voice (default 30) |
| `--out DIR` | Output directory (default `samples`) |
| `--max-seconds S` | Discard clips longer than this (default 2.5) – catches misreadings |

Run `python gen_samples.py --help` for everything.

### Tips

- **Many voices beat many clips per voice.** 40 speakers × 60 clips teach more
  than 5 speakers × 500.
- **Listen before generating thousands.** Some voices read a foreign-looking name
  with their own spelling rules (a German voice may say "Iarvis"). A phonetic
  spelling (`--phrase "Hey Dscharwis"`) can fix that, but native speakers who
  simply get it right are the better source.
- Keep ElevenLabs on a flash/turbo v2.5 or v3 model – only those accept a
  language code.

## Recording your own voice

Record one voice memo per person, saying the phrase 20–50 times with ~2 seconds of
silence in between. Vary it: normal, fast, quiet, casual, near and far, in the
rooms where the devices sit.

```bash
python split_recording.py "recordings/raw/alice*" --name alice
```

Keep the quotes around wildcard paths: PowerShell does not expand `*` for
Python, so the script does it itself – and in bash the quotes stop the shell
from expanding it first.

Clips go to `recordings/alice/`. Pieces that are too short or too long land in
`recordings/alice/_check/` for review. If detection is off, tune
`--min-silence` (ms) or `--thresh` (dB below average level).

Even 20–50 real clips per person measurably improve recognition – they are the
strongest lever there is.

## Using the clips with oww_forge

Either upload them in the forge's web UI (**Improve → Add recordings…**), or copy
them into the wake word's clip directories, keeping roughly 10 % for testing.
The upload is easiest for a few dozen recordings; for thousands of generated
clips, copying is more reliable.

The forge runs in Docker, so its `data/` directory lives wherever you cloned
EchoMuse. **On Windows that is usually inside WSL2**, and the copy commands
below run in the **WSL shell**, not in PowerShell:

- Windows paths are reachable from WSL under `/mnt/c/...`
  (e.g. `C:\wakeword-samples` → `/mnt/c/wakeword-samples`).
- The other way round, open a WSL folder in Explorer with `explorer.exe .`, or
  via `\\wsl.localhost\<distro>\home\<user>\...` (list distros with `wsl -l`).
- The container creates its directories as root, hence `sudo` below.

On Linux, run the same commands in any shell.

```bash
SRC=/mnt/c/path/to/wakeword-samples/samples/eleven   # Linux: /path/to/samples/eleven
DST=~/EchoMuse/oww_forge/data/wakewords/<name>/<name>
sudo mkdir -p "$DST/positive_train" "$DST/positive_test"

ls "$SRC"/*.wav | shuf > /tmp/list.txt
n=$(( $(wc -l < /tmp/list.txt) / 10 ))
head -n "$n" /tmp/list.txt | sudo xargs -d '\n' cp -t "$DST/positive_test/"
tail -n +"$((n + 1))" /tmp/list.txt | sudo xargs -d '\n' cp -t "$DST/positive_train/"
```

Don't do both, or every clip ends up twice. The forge counts existing clips
toward its sample target and fills the rest with its American voices, so the
share of native clips is `your clips / Example voices` – lower *Example voices*
to raise it.
