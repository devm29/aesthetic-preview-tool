"""Streamlit front end.

This file does presentation only: widgets, layout and copy. Every pixel of
image work happens in ``src.pipeline`` and the treatment registry, which is
what makes the pipeline testable without a browser.
"""

from __future__ import annotations

import json
from html import escape

import numpy as np
import streamlit as st

from src.config import DISCLAIMER, get_settings
from src.detection import available_detectors
from src.generative import build_editor
from src.identity import EXPLANATION as IDENTITY_EXPLANATION
from src.imaging import (
    ImageTooLargeError,
    UnreadableImageError,
    encode_png,
    image_fingerprint,
    wipe_composite,
)
from src.masks import skin_mask
from src.models import QCReport, TreatmentContext
from src.narrative import narrator_available, summarise
from src.pipeline import (
    analyse,
    build_export,
    describe_result,
    load_preview_image,
    render_preview,
    treatment_availability,
)
from src.treatments import ADJUSTMENT, PROCEDURE

SETTINGS = get_settings()

st.set_page_config(
    page_title="Aesthetic Preview",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="auto",
)

# Linen, ink and clay: a consulting-room palette rather than a dashboard one.
# Fraunces / Public Sans / IBM Plex Mono are loaded by .streamlit/config.toml;
# the stacks fall back to system fonts when offline.
STYLE = """
<style>
:root {
  --ink:#23201c; --muted:#5b544c; --line:#e2dad0; --surface:#ffffff; --wash:#f3eee7;
  --clay:#94452b; --pass:#2d6a4c; --fail:#a1372a; --skip:#6a645c;
  --serif:"Fraunces", Georgia, "Times New Roman", serif;
  --mono:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
.block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1320px; }
h1, h2, h3, h4 { color:var(--ink); }
.masthead { display:flex; align-items:baseline; gap:.5rem 1rem; flex-wrap:wrap;
            border-bottom:1px solid var(--line); padding-bottom:.7rem; margin-bottom:.8rem; }
.masthead h1 { font-family:var(--serif); font-weight:600; font-size:2rem; margin:0; padding:0;
               letter-spacing:-.015em; }
.masthead .kicker { font-size:.72rem; font-weight:600; letter-spacing:.12em;
                    text-transform:uppercase; color:var(--clay); }
.lede { color:var(--muted); font-size:.98rem; margin:0 0 .9rem; max-width:72ch; line-height:1.55; }
.notice { border-left:3px solid var(--clay); background:var(--wash); color:var(--ink);
          border-radius:0 8px 8px 0; padding:.65rem .95rem; font-size:.86rem; line-height:1.55;
          margin-bottom:1.1rem; }
.section-title { font-family:var(--serif); font-weight:600; font-size:1.4rem !important;
                 margin:1.6rem 0 .7rem !important; padding:0 !important; color:var(--ink); }
.qc-grid { list-style:none; padding:0 !important; margin:0 0 1rem !important; display:grid;
           gap:.6rem; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); }
.qc { border:1px solid var(--line); border-radius:8px; padding:.65rem .8rem;
      background:var(--surface); margin:0 !important; }
.qc .head { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
.qc .name { font-size:.78rem; font-weight:600; color:var(--ink); }
.qc .status { font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
              padding:.1rem .45rem; border-radius:4px; white-space:nowrap; }
.qc.pass .status { color:var(--pass); background:#e6f0ea; }
.qc.fail .status { color:var(--fail); background:#f7e4e0; }
.qc.skipped .status { color:var(--skip); background:#ece8e2; }
.qc.fail { border-color:#d9a79e; }
.qc .val { font-family:var(--mono); font-size:1.1rem; color:var(--ink); margin-top:.3rem; }
.qc .val small { font-family:inherit; font-size:.72rem; color:var(--muted); }
.qc .det { font-size:.78rem; color:var(--muted); margin-top:.2rem; line-height:1.4; }
.panel { border:1px solid var(--line); border-radius:8px; padding:.95rem 1.1rem;
         background:var(--surface); height:100%; }
.panel h3 { font-family:var(--serif); font-weight:600; margin:0 0 .4rem; padding:0;
            font-size:1.02rem; }
.panel p, .panel li { margin:0; font-size:.85rem; color:var(--muted); line-height:1.55; }
.panel ul { margin:0; padding-left:1.1rem; }
.panel .explain { margin-top:.7rem; padding-top:.6rem; border-top:1px dashed var(--line); }
.score { font-family:var(--mono); font-size:2rem; color:var(--ink); line-height:1.1; }
.score small { font-size:.78rem; color:var(--muted); }
.verdict { margin-top:.35rem; font-weight:600; color:var(--ink) !important; }
.footer { color:var(--muted); font-size:.8rem; border-top:1px solid var(--line);
          margin-top:2rem; padding-top:.9rem; line-height:1.6; }
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
[data-testid="stImageCaption"] { color:var(--muted); font-size:.8rem; }
:focus-visible { outline:2px solid var(--clay) !important; outline-offset:2px; }
@media (max-width: 640px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:3.6rem; }
  .masthead h1 { font-size:1.6rem; }
  .qc-grid { grid-template-columns:1fr 1fr; }
  .score { font-size:1.7rem; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------- caching
# Streamlit reruns this script on every widget change. Without these caches the
# app would re-decode, re-detect and re-mask the face on every slider nudge.
@st.cache_data(show_spinner=False, max_entries=8)
def cached_image(data: bytes) -> np.ndarray:
    return load_preview_image(data, SETTINGS)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_report(fingerprint: str, _image: np.ndarray) -> QCReport:
    return analyse(_image, SETTINGS)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_skin_mask(fingerprint: str, _image: np.ndarray, _report: QCReport) -> np.ndarray:
    return skin_mask(_image, _report.face)


@st.cache_resource(show_spinner=False)
def cached_editor():
    return build_editor(SETTINGS)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_preview(fingerprint: str, intensities: tuple[tuple[str, int], ...], _image, _report):
    context = TreatmentContext(
        original=_image,
        geometry=_report.face,
        skin_mask=cached_skin_mask(fingerprint, _image, _report),
        settings=SETTINGS,
        editor=cached_editor(),
    )
    return render_preview(
        _image, _report, dict(intensities), SETTINGS, context.editor, context=context
    )


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Photo")
    upload = st.file_uploader("Upload a portrait", type=["jpg", "jpeg", "png"])
    use_sample = st.toggle(
        "Use the synthetic sample",
        value=upload is None,
        help="A procedurally generated face (tools/make_sample_face.py). Not a real person.",
    )
    st.caption(f"Uploads are processed in memory only. Limit {SETTINGS.max_upload_mb} MB.")

    st.divider()
    st.subheader("Engine")
    detectors = available_detectors()
    st.caption("Face detector: " + (", ".join(detectors) if detectors else "none available"))
    editor = cached_editor()
    if editor.available():
        st.caption(f"Generative engine: {editor.name}")
    else:
        st.caption("Generative engine: off. " + editor.unavailable_reason())
    st.caption("Narrator: Claude" if narrator_available(SETTINGS) else "Narrator: offline template")

    st.divider()
    force = st.checkbox("Render even if checks fail", value=False)


# --------------------------------------------------------------------------- header
st.markdown(
    '<header class="masthead"><h1>Aesthetic Preview</h1>'
    '<span class="kicker">Proof of concept &middot; runs locally on CPU</span></header>'
    '<p class="lede">See what common non-invasive cosmetic adjustments would look like as '
    "an image edit, with the original photo always beside the result.</p>"
    f'<aside class="notice" role="note"><b>Read this first.</b> {escape(DISCLAIMER)}</aside>',
    unsafe_allow_html=True,
)


def section(title: str) -> None:
    st.markdown(f'<h2 class="section-title">{escape(title)}</h2>', unsafe_allow_html=True)


def _load_source() -> bytes:
    if upload is not None and not use_sample:
        return upload.getvalue()
    path = SETTINGS.sample_image_path
    if not path.is_file():
        st.error(f"Sample image missing at {path}. Run: python tools/make_sample_face.py")
        st.stop()
    return path.read_bytes()


try:
    image = cached_image(_load_source())
except (ImageTooLargeError, UnreadableImageError) as exc:
    st.error(str(exc))
    st.stop()

fingerprint = image_fingerprint(image)
report = cached_report(fingerprint, image)

# --------------------------------------------------------------------------- QC
# The status is spelled out on every card, so colour is never the only signal.
STATUS_WORD = {"pass": "Pass", "fail": "Fail", "skipped": "Not measured"}

section("Photo quality")
cards = []
for check in report.checks:
    if check.value is None:
        reading = ""
    else:
        unit = f" <small>{escape(check.unit)}</small>" if check.unit else ""
        reading = f'<div class="val">{check.value:.1f}{unit}</div>'
    cards.append(
        f'<li class="qc {check.status}"><div class="head">'
        f'<span class="name">{escape(check.name)}</span>'
        f'<span class="status">{STATUS_WORD[check.status]}</span></div>'
        f'{reading}<div class="det">{escape(check.detail)}</div></li>'
    )
st.markdown(
    f'<ul class="qc-grid" aria-label="Photo quality checks">{"".join(cards)}</ul>',
    unsafe_allow_html=True,
)

if report.face is None:
    st.error(
        "No face was found, so nothing can be previewed. Use a frontal photo with the "
        "face well lit and filling a good part of the frame."
    )
    st.image(image, caption="The photo as uploaded", width=420)
    st.stop()
if report.failures and not force:
    st.warning(
        "Some checks failed. Retake the photo, or tick "
        "**Render even if checks fail** in the sidebar to continue anyway."
    )
    st.image(image, caption="The photo as uploaded", width=420)
    st.stop()

# --------------------------------------------------------------------------- controls
context_probe = TreatmentContext(
    original=image,
    geometry=report.face,
    skin_mask=cached_skin_mask(fingerprint, image, report),
    settings=SETTINGS,
    editor=cached_editor(),
)

section("Treatments")
intensities: dict[str, int] = {}
for group, heading in ((ADJUSTMENT, "Skin adjustments"), (PROCEDURE, "Procedure previews")):
    rows = treatment_availability(context_probe, group)
    if not rows:
        continue
    st.caption(heading)
    columns = st.columns(max(len(rows), 1))
    for column, (treatment, usable, reason) in zip(columns, rows, strict=False):
        with column:
            intensities[treatment.key] = st.slider(
                treatment.spec.label,
                0,
                treatment.spec.max_intensity,
                treatment.spec.default if usable else 0,
                step=treatment.spec.step,
                help=treatment.spec.summary if usable else reason,
                disabled=not usable,
                key=f"slider_{treatment.key}",
            )
            if usable:
                st.caption(f"{treatment.engine(context_probe)} engine")
            else:
                st.caption(f"Unavailable: {reason.split('.')[0].lower()}")

with st.spinner("Rendering the preview..."):
    preview = cached_preview(fingerprint, tuple(sorted(intensities.items())), image, report)

# --------------------------------------------------------------------------- compare
section("Before and after")
mode = st.segmented_control("Comparison view", ["Side by side", "Wipe"], default="Side by side")
if mode == "Wipe":
    position = st.slider(
        "Wipe position",
        0.0,
        1.0,
        0.5,
        0.01,
        help="0 shows only the preview, 1 only the original.",
    )
    # One image at full width would be twice the size of each side-by-side pane.
    _, wipe_col, _ = st.columns([1, 3, 1])
    with wipe_col:
        st.image(
            wipe_composite(preview.original, preview.edited, position),
            caption="Left of the line is the original photo; right is the preview.",
            width="stretch",
        )
else:
    left, right = st.columns(2)
    with left:
        st.image(preview.original, caption="Original photo", width="stretch")
    with right:
        st.image(preview.edited, caption="Preview, edited", width="stretch")

# --------------------------------------------------------------------------- signals
identity = preview.identity
signal_col, detail_col = st.columns([1, 2])
with signal_col:
    if identity is not None:
        st.markdown(
            f'<section class="panel" aria-label="Identity signal"><h3>Identity signal</h3>'
            f'<div class="score">{identity.ssim:.3f} '
            f"<small>SSIM, threshold {identity.threshold:.2f}</small></div>"
            f'<p class="verdict">{escape(identity.verdict)}.</p></section>',
            unsafe_allow_html=True,
        )
        if not identity.within_threshold:
            st.warning(
                "This edit changed the face structure substantially. Compare it against "
                "the original carefully before reading anything into it."
            )
with detail_col:
    if preview.applied:
        items = "".join(
            f"<li>{escape(a.label)} at {a.intensity}% "
            f"({escape(a.engine)}, {a.elapsed_ms:.0f} ms)</li>"
            for a in preview.applied
        )
        applied = f"<ul>{items}</ul>"
    else:
        applied = "<p>Nothing applied yet. Move a slider above to see an edit.</p>"
    st.markdown(
        f'<section class="panel" aria-label="What was applied"><h3>What was applied</h3>'
        f'{applied}<p class="explain">{escape(IDENTITY_EXPLANATION)}</p></section>',
        unsafe_allow_html=True,
    )

for note in preview.notes:
    st.info(note)

# --------------------------------------------------------------------------- notes + export
summary = describe_result(preview, report)

with st.expander("Consultation notes", expanded=False):
    st.caption(
        "Plain-language notes you could take to a practitioner. Generated by Claude when "
        "ANTHROPIC_API_KEY is set, otherwise written from a fixed template."
    )
    # Notes describe one specific edit. Keyed on the photo and the slider
    # positions, so moving a slider hides notes written for the previous edit.
    notes_key = (fingerprint, tuple(sorted(intensities.items())))
    if st.button("Write the notes", type="secondary", key="write_notes"):
        with st.spinner("Writing..."):
            st.session_state["narrative"] = (notes_key, summarise(summary, SETTINGS))
    stored = st.session_state.get("narrative")
    narrative = stored[1] if stored is not None and stored[0] == notes_key else None
    if narrative is not None:
        st.markdown(narrative.text)
        st.caption(
            f"Source: {narrative.source}." + (f" {narrative.detail}" if narrative.detail else "")
        )

export_col, json_col, _ = st.columns([1, 1, 2])
with export_col:
    st.download_button(
        "Download before/after",
        data=encode_png(build_export(preview, SETTINGS)),
        file_name=f"preview_{fingerprint}.png",
        mime="image/png",
        help="The export always contains the original beside the edit, plus a disclaimer.",
        width="stretch",
    )
with json_col:
    st.download_button(
        "Download summary (JSON)",
        data=json.dumps(summary, indent=2),
        file_name=f"preview_{fingerprint}.json",
        mime="application/json",
        width="stretch",
    )

st.markdown(
    f'<footer class="footer">{escape(DISCLAIMER)} Nothing here is medical advice, and no '
    "image leaves this machine: detection, editing and scoring all run locally on CPU.</footer>",
    unsafe_allow_html=True,
)
