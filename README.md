# Aesthetic Preview (Derma-Scan)

A local, CPU-only **proof of concept — not a medical device** — that previews what common
non-invasive cosmetic adjustments would look like *as an image edit* on a portrait. It
quality-checks the photo, applies filters and geometric warps inside a detected face
region, keeps the original on screen beside the result, and reports an SSIM signal for how
much image structure the edit changed.

> **Read this first.** The previews are illustrative image edits: they do not model biology
> and are not a simulation, a prediction or a guarantee of any clinical outcome, and
> nothing here is medical advice. No image leaves the machine — detection, editing and
> scoring all run locally on CPU.

## There is no real person's face in this repository

`assets/sample_face.jpg` is **procedurally generated** by `tools/make_sample_face.py`: a
few hundred seeded OpenCV primitives, `SEED = 20260923`, 768 × 768, no photographic
source. It reproduces byte for byte, so the claim is yours to check:

```bash
python tools/make_sample_face.py && md5sum assets/sample_face.jpg  # 66443d667af2e07d...
```

Every screenshot below uses it, the sidebar toggle that loads it says *"A procedurally
generated face … Not a real person"*, and no test touches a real photograph.

## What the screens look like

All of these come from the Docker image (`docker compose up --build`), which ships no
optional ML packages: Haar detection only, so the landmark procedures are disabled *with
the reason shown* and head tilt reads "Not measured".

**Photo quality, then controls.** Every check names its own verdict in words, so a check
the detector cannot measure says so rather than quietly passing.

![Photo-quality report and treatment controls](docs/screenshots/01-overview.png)

**Before and after.** The original is never off screen, each treatment is itemised with
its engine and cost, and a [wipe view](docs/screenshots/03-wipe-compare.png) is the
alternative.

![Side-by-side comparison, identity signal and applied treatments](docs/screenshots/02-compare-and-identity.png)

**Consultation notes.** What the edit did and what the identity number does not mean,
written by Claude when the narrator is installed and a key is set, otherwise by the
offline template shown here. Notes belong to one edit: move a slider and they are hidden
until you ask again.

![Consultation notes from the offline template](docs/screenshots/04-consultation-notes.png)

**A file that is not an image** gets a sentence, not a stack trace; so does an upload over
the size or pixel limit.

![Error shown for an unreadable upload](docs/screenshots/05-unreadable-upload.png)

**On a phone** (390 px wide) the sidebar starts collapsed and the panels stack.

<p>
  <img src="docs/screenshots/06-mobile-overview.png" alt="Quality checks on a 390 px wide screen" width="260">
  <img src="docs/screenshots/07-mobile-compare.png" alt="Before and after on a 390 px wide screen" width="260">
</p>

## From upload to preview

`src/pipeline.py` is the only module that knows this order and it imports no Streamlit
(`test_the_pipeline_never_imports_streamlit` pins that): `app.py` renders widgets and
copy, `src/cli.py` drives the identical path headlessly.

```mermaid
flowchart TD
    IN["Upload, max 12 MB, or the synthetic sample"] --> PREP["decode_image then downscale<br/>to DERMA_PREVIEW_MAX_SIDE"]
    PREP --> DET["detect_face: MediaPipe FaceMesh<br/>if installed, else Haar bounding box"]
    DET --> QC{"run_qc: face present, face size,<br/>sharpness, lighting, head tilt"}
    QC -->|"no face, or a failed<br/>check with no override"| HALT["Show the photo and every check's<br/>own verdict. Nothing is edited."]
    QC -->|"clear, or overridden"| CTX["build_context:<br/>skin mask, geometry, editor"]
    CTX --> ADJ["Adjustments, masked to skin:<br/>wrinkle, tone, pigmentation"]
    ADJ --> PROC{"A procedure slider is up<br/>and landmarks exist?"}
    PROC -->|"no landmarks: disabled,<br/>reason noted"| ID
    PROC -->|"yes"| WARP["Parametric lip-ring warp and under-eye<br/>chroma, or SDXL inpaint if enabled"]
    WARP --> ID["compute_identity: SSIM over the<br/>face crop, before versus after"]
    ID --> OUT["Compare, notes, JSON summary,<br/>disclaimer-stamped export"]
```

Treatments run in registry order and each is independently survivable: one whose region
cannot be located, whose dependency is missing, or which raises, is skipped with a note
the user sees (`test_a_treatment_that_raises_is_skipped_rather_than_crashing_the_app`).

## Pass, fail, or "not measured"

Quality control is per-check, and `skipped` is a first-class status: the Haar fallback has
no landmarks, so `FaceGeometry.roll_deg` is `None` and the tilt check reports `skipped`
rather than inventing a compliant `0.0°`
(`test_check_orientation_is_skipped_when_roll_cannot_be_measured`).

```mermaid
stateDiagram-v2
    [*] --> Measuring : a face was detected
    Measuring --> Pass : value inside the threshold
    Measuring --> Fail : value outside the threshold
    Measuring --> Skipped : the input is missing, e.g. no landmarks means no roll
    Fail --> Halted : any failure stops rendering
    Halted --> Rendering : the user ticks Render even if checks fail
    Pass --> Rendering
    Skipped --> Rendering : never a pass and never a failure
    Rendering --> [*]
```

Every threshold lives in `QCThresholds` (`src/config.py`): face coverage ≥ 4 % of the
frame, Laplacian variance ≥ 100 at a fixed 512 px width so the score is size-stable, mean
brightness 90–200, contrast ≥ 25, ≤ 15 % clipped highlights or shadows, roll within ±20°.
They are heuristics tuned on the synthetic sample, which is why they are configurable
rather than scattered through `src/qc.py`.

## The identity signal, and what it is not

SSIM on the greyscale face crop, original versus edit, reported against
`DERMA_IDENTITY_MIN_SSIM` (default `0.80`) with a plain verdict: *Barely changed* above
0.95, *Visible change, structure retained* above the threshold, *Large structural change*
below it. A crop too small for an SSIM window reports `0.0`, not a reassuring number.

It says how much image **structure** survived — nothing more. It is not a biometric match,
not identity verification, and a high score does not make an edit accurate, appropriate or
safe: wording that ships in the UI, the CLI, the JSON export and the narrator's system
prompt, and that `test_the_explanation_disclaims_biometric_identity` keeps there.

Two related guard-rails: no code path exports a lone "after" — `build_export` writes the
original beside the edit with a burnt-in banner
(`test_build_export_puts_the_original_next_to_the_edit`) — and the narrator's prompt forbids
clinical claims, dose or product recommendations and any comment on the person's appearance,
falling back to the template on refusal, truncation or any API error.

## Running it

```bash
docker compose up --build   # then open http://localhost:8200
docker compose down -v
```

It boots with the synthetic portrait already loaded, so there is nothing to upload first.
The image ships the **lightweight path**: `python:3.12-slim`, multi-stage, non-root
(`derma`, uid 10001), read-only root filesystem with tmpfs mounts, `no-new-privileges`,
healthchecked on `/_stcore/health`. The current build is about 296 MB compressed
(1.27 GB unpacked, most of it OpenCV, SciPy and PyArrow). Inside it `mediapipe`, `torch`,
`diffusers` and `anthropic` are absent and `available_detectors()` returns `['haar']`,
so "works with no ML dependencies" is the path actually built and booted.

To let Claude write the consultation notes, build with the narrator SDK and pass a key
(copy `.env.example` to `.env` and fill in the last two lines, or export them):

```bash
DERMA_WITH_NARRATOR=true ANTHROPIC_API_KEY=... docker compose up --build
```

Locally, on Python 3.12 (`.python-version`), plus the optional extras the app detects at
runtime and names in the sidebar:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && streamlit run app.py   # http://localhost:8501

pip install -r requirements-landmarks.txt  # MediaPipe FaceMesh -> lip and under-eye previews
pip install -r requirements-ml.txt         # SDXL inpainting; also set DERMA_ENABLE_GENERATIVE=true
pip install -r requirements-narrator.txt   # Claude-written notes; also set ANTHROPIC_API_KEY
```

## The same pipeline, without a browser

`python -m src.cli` walks the identical code path, grows one flag per registered treatment,
and is the proof that the image layer has no UI dependency. A recorded run on the sample:

```console
$ python -m src.cli assets/sample_face.jpg --wrinkle 55 --tone 30
Image      : assets/sample_face.jpg (768x768)
  [PASS] Face detected: One face located by the haar detector.
  [PASS] Face size (30.9 % of frame): Face fills enough of the frame.
  [PASS] Sharpness (141.5 Laplacian variance): Detail looks sufficient.
  [PASS] Lighting (130.2 mean brightness): Even light (contrast 43).
  [SKIP] Head tilt: The haar detector does not provide landmarks, so head tilt could not be measured.
Detector   : haar
  applied  Wrinkle smoothing @ 55% (parametric, 11 ms)
  applied  Tone evening @ 30% (parametric, 101 ms)
  applied  Pigmentation evening @ 20% (parametric, 12 ms)
Identity   : SSIM 0.979 (threshold 0.80) - Barely changed
Total      : 141 ms

This is a proof of concept, not a medical device. Previews are illustrative image edits, not a prediction or guarantee of any clinical outcome.
```

`--out preview.png` writes the before/after pair, `--json` prints the structure the UI
downloads and the narrator reads, and `--force` renders past failed checks. A missing or
unreadable file exits with status 2 and a one-line reason; a photo with no face, or one
that fails a check without `--force`, exits with 1.

The start of the `--json` output for the same run:

```json
{
  "disclaimer": "This is a proof of concept, not a medical device. ...",
  "detector": "haar",
  "landmarks": false,
  "quality_checks": [
    {"name": "Face detected", "status": "pass", "value": null, "detail": "One face located by the haar detector."},
    {"name": "Face size", "status": "pass", "value": 30.91244167751736, "detail": "Face fills enough of the frame."},
    ...
    {"name": "Head tilt", "status": "skipped", "value": null, "detail": "The haar detector does not provide landmarks, so head tilt could not be measured."}
  ],
  "treatments": [{"key": "wrinkle", "label": "Wrinkle smoothing", "intensity": 55, "engine": "parametric", ...}, ...],
  "identity": {"ssim": 0.979, "threshold": 0.8, "within_threshold": true, "verdict": "Barely changed"},
  ...
}
```

## Settings

`src/config.py` resolves these once at startup and is the only place the app reads its own
settings from `os.environ` (`.env.example` lists them all); the one other reader is `src/narrative.py`, which checks whether
an Anthropic credential is present before attempting a call.

| Variable | Default | What it does |
| --- | --- | --- |
| `DERMA_PREVIEW_MAX_SIDE` | `1024` | Longest side an image is downscaled to before any processing. Cost is quadratic in it, so it is the main latency lever. |
| `DERMA_MAX_UPLOAD_MB` | `12` | Upload ceiling, enforced in `imaging.decode_image`. |
| `DERMA_IDENTITY_MIN_SSIM` | `0.80` | Threshold the identity signal is reported against. |
| `DERMA_FACE_DETECTOR` | `auto` | `auto`, `mediapipe` or `haar`; an unavailable preference falls back with a warning. |
| `DERMA_ENABLE_GENERATIVE` | `false` | Enables SDXL inpainting. Needs `requirements-ml.txt`; without it the UI says so and uses the parametric engines. |
| `DERMA_NARRATOR_ENABLED` | `true` | Allows Claude-written consultation notes. Off means the fixed template is always used. |
| `ANTHROPIC_API_KEY` | — | **Unset is a supported state**: the notes come from the deterministic template instead. Also needs `requirements-narrator.txt`. |
| `DERMA_WITH_NARRATOR` | `false` | Compose build argument, not read by the app: `true` installs the narrator SDK into the image. |

Also read: `DERMA_EXPORT_MAX_SIDE` (`2048`), `DERMA_NARRATOR_MODEL` (`claude-opus-5`), `DERMA_LOG_LEVEL` (`INFO`), `DERMA_SDXL_MODEL_ID`, `DERMA_SDXL_MAX_SIDE` (`768`).

## Adding a treatment

One class. Its `TreatmentSpec` carries everything a UI needs to render a control — label,
help text, range, step, ordering, landmark and generative requirements — so no surface
grows a per-treatment branch:

```python
from src.treatments import ADJUSTMENT, Treatment, TreatmentSpec, register


@register
class Sepia(Treatment):
    spec = TreatmentSpec(
        key="sepia",
        label="Sepia",
        summary="Warms the skin region.",
        group=ADJUSTMENT,
        order=5,
        default=0,
    )

    def apply(self, image_rgb, context, intensity):
        return image_rgb  # must be a no-op at intensity 0
```

Import it in `src/treatments/__init__.py` and it becomes a slider, a `--sepia` flag, a row
in the JSON summary and an input to the consultation notes — nothing else changes, and
`test_a_third_party_treatment_needs_only_a_class` registers exactly this class at test time
to prove it. Detectors and the generative editor use the same shape — a `Protocol` with
`available()`, `unavailable_reason()` and the work method — so adding YuNet or RetinaFace is
one class plus `register_detector`.

## Tests and tooling

```bash
pip install -r requirements-dev.txt
pytest                                  # 210 tests across 12 files
ruff check . && ruff format --check .
```

No test downloads a model, needs a GPU, makes a network call or touches a real photograph:
`tests/conftest.py` supplies a synthetic FaceMesh landmark array and a recording generative
editor, `tests/test_narrative.py` installs a fake `anthropic` module, `tests/test_generative.py`
substitutes the SDXL pipeline, and the suite passes with none of MediaPipe, torch or `anthropic`
installed. `tests/test_app.py` drives the Streamlit page itself through `streamlit.testing`:
it boots on the sample, checks the landmark sliders are disabled on a Haar face, and moves a
slider after writing notes to prove the stale notes disappear.

Streamlit re-runs the whole script on every slider nudge, so decode, detection and the skin
mask are cached on a SHA-256 fingerprint of the pixels, and `imaging.masked_filter` crops to
the mask's bounding box before the expensive bilateral/CLAHE pass —
`test_masked_filter_matches_a_full_frame_pass_for_a_shift_invariant_filter` proves that crop
changes nothing.

## What it does not do

- **Not a medical device, and not a biological simulation.** Filters and warps cannot
  predict how tissue responds to a procedure; nothing here should inform a real decision.
- **The identity number is change magnitude, not identity verification.** It will not
  catch an edit that is structurally small but perceptually wrong.
- **Lip and under-eye previews need MediaPipe FaceMesh.** Without it the Haar cascade
  gives a bounding box only, those treatments are disabled and the skin mask degrades to an
  inset ellipse. The app uses the legacy `mediapipe.solutions.face_mesh` API, which builds
  from 0.10.30 onward drop on some platforms — hence the cap in
  `requirements-landmarks.txt` and an `available()` that probes the API, not the import.
- **Generative editing is slow and untested end to end.** SDXL inpainting on CPU takes
  minutes per pass; dispatch, ROI cropping and compositing are covered with a fake pipeline,
  but real model output is not — no test may download weights.
- **One face per image**, frontal and reasonably lit. No multi-face handling, no pose
  correction, no profiles.
- **No accounts and no persistence.** Images are processed in memory and discarded; nothing
  is written to disk except what you choose to download.

## License

MIT — see [LICENSE](LICENSE).
