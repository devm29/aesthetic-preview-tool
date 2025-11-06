"""Streamlit front end.

This file does presentation only: widgets, layout and copy. Every pixel of
image work happens in ``src.pipeline`` and the treatment registry, which is
what makes the pipeline testable without a browser.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from src.config import DISCLAIMER, get_settings
from src.detection import available_detectors
from src.generative import build_editor
from src.identity import EXPLANATION as IDENTITY_EXPLANATION
from src.imaging import ImageTooLargeError, encode_png, image_fingerprint, wipe_composite
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
    page_title="Aesthetic Preview POC",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
:root { --ink:#1d2430; --muted:#5d6b7f; --line:#e3e8ef; --accent:#3f6d8e; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1360px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
.hero { display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap; margin-bottom:.15rem; }
.hero h1 { font-size:1.85rem; margin:0; }
.tag { font-size:.68rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
       padding:.22rem .55rem; border-radius:999px; background:#eef3f8; color:var(--accent);
       border:1px solid #d5e2ed; }
.lede { color:var(--muted); font-size:.95rem; margin:.1rem 0 .8rem; max-width:78ch; }
.notice { border:1px solid #e7d9b8; background:#fdf8ec; color:#6b5726; border-radius:10px;
          padding:.65rem .9rem; font-size:.85rem; line-height:1.5; margin-bottom:.9rem; }
.qc-grid { display:flex; gap:.6rem; flex-wrap:wrap; margin:.2rem 0 .9rem; }
.qc { flex:1 1 170px; border:1px solid var(--line); border-radius:10px; padding:.6rem .75rem;
      background:#fff; }
.qc .name { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); }
.qc .val { font-size:1.15rem; font-weight:650; color:var(--ink); margin-top:.1rem; }
.qc .det { font-size:.74rem; color:var(--muted); margin-top:.25rem; line-height:1.35; }
.qc.pass { border-left:3px solid #3f8e63; }
.qc.fail { border-left:3px solid #b5523f; }
.qc.skipped { border-left:3px solid #9aa6b5; }
.panel { border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; background:#fff; }
.panel h4 { margin:0 0 .35rem; font-size:.95rem; color:var(--ink); }
.panel p { margin:0; font-size:.82rem; color:var(--muted); line-height:1.55; }
.score { font-size:2.1rem; font-weight:680; color:var(--ink); line-height:1.1; }
.score small { font-size:.8rem; font-weight:500; color:var(--muted); }
.footer { color:var(--muted); font-size:.78rem; border-top:1px solid var(--line);
          margin-top:2rem; padding-top:.9rem; line-height:1.6; }
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
h4 { margin-top:1.1rem !important; margin-bottom:.35rem !important; }
[data-testid="stSlider"] { padding-bottom:0; }
[data-testid="stSlider"] + [data-testid="stCaptionContainer"] { margin-top:-.35rem; }
[data-testid="stImageCaption"] { color:var(--muted); font-size:.78rem; }
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
    st.markdown("### Photo")
    upload = st.file_uploader("Upload a portrait", type=["jpg", "jpeg", "png"])
    use_sample = st.toggle(
        "Use the synthetic sample",
        value=upload is None,
        help="A procedurally generated face (tools/make_sample_face.py). Not a real person.",
    )
    st.caption(f"Uploads are processed in memory only. Limit {SETTINGS.max_upload_mb} MB.")

    st.divider()
    st.markdown("### Engine")
    detectors = available_detectors()
    st.caption("Face detector: " + (", ".join(detectors) if detectors else "none available"))
    editor = cached_editor()
    if editor.available():
        st.caption(f"Generative engine: {editor.name}")
    else:
        st.caption("Generative engine: off - " + editor.unavailable_reason())
    st.caption("Narrator: Claude" if narrator_available(SETTINGS) else "Narrator: offline template")

    st.divider()
    force = st.checkbox("Render even if checks fail", value=False)


# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>Aesthetic Preview</h1>'
    '<span class="tag">Proof of concept</span>'
    '<span class="tag">Local &amp; CPU only</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="lede">Preview what common non-invasive cosmetic adjustments would look '
    "like as an image edit, with the original always shown beside the result.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="notice"><b>Read this first.</b> {DISCLAIMER}</div>',
    unsafe_allow_html=True,
)


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
except ImageTooLargeError as exc:
    st.error(str(exc))
    st.stop()

fingerprint = image_fingerprint(image)
report = cached_report(fingerprint, image)

# --------------------------------------------------------------------------- QC
STATUS_WORD = {"pass": "Pass", "fail": "Fail", "skipped": "Not measured"}

st.markdown("#### Photo quality")
cards = []
for check in report.checks:
    # A check with no number still needs a headline, or the card reads as broken.
    if check.value is None:
        value, unit = STATUS_WORD[check.status], ""
    else:
        value = f"{check.value:.1f}"
        unit = f' <small style="color:#5d6b7f">{check.unit}</small>' if check.unit else ""
    cards.append(
        f'<div class="qc {check.status}"><div class="name">{check.name}</div>'
        f'<div class="val">{value}{unit}</div>'
        f'<div class="det">{check.detail}</div></div>'
    )
st.markdown(f'<div class="qc-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

if report.face is None:
    st.image(image, caption="Uploaded photo", width=420)
    st.stop()
if report.failures and not force:
    st.warning(
        "Some checks failed. Retake the photo, or tick "
        "**Render even if checks fail** in the sidebar to continue anyway."
    )
    st.image(image, caption="Uploaded photo", width=420)
    st.stop()

# --------------------------------------------------------------------------- controls
context_probe = TreatmentContext(
    original=image,
    geometry=report.face,
    skin_mask=cached_skin_mask(fingerprint, image, report),
    settings=SETTINGS,
    editor=cached_editor(),
)

st.markdown("#### Treatments")
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
                st.caption(f"Unavailable - {reason.split('.')[0].lower()}")

preview = cached_preview(fingerprint, tuple(sorted(intensities.items())), image, report)

# --------------------------------------------------------------------------- compare
st.markdown("#### Before and after")
mode = st.segmented_control(
    "Comparison", ["Side by side", "Wipe"], default="Side by side", label_visibility="collapsed"
)
if mode == "Wipe":
    position = st.slider("Wipe position", 0.0, 1.0, 0.5, 0.01, label_visibility="collapsed")
    st.image(
        wipe_composite(preview.original, preview.edited, position),
        caption="Left of the line is the original photo; right is the preview.",
        width="stretch",
    )
else:
    left, right = st.columns(2)
    with left:
        st.image(preview.original, caption="Original", width="stretch")
    with right:
        st.image(preview.edited, caption="Preview (edited)", width="stretch")

# --------------------------------------------------------------------------- signals
identity = preview.identity
signal_col, detail_col = st.columns([1, 2])
with signal_col:
    if identity is not None:
        st.markdown(
            f'<div class="panel"><h4>Identity signal</h4>'
            f'<div class="score">{identity.ssim:.3f} '
            f"<small>/ threshold {identity.threshold:.2f}</small></div>"
            f'<p style="margin-top:.35rem">{identity.verdict}.</p></div>',
            unsafe_allow_html=True,
        )
        if not identity.within_threshold:
            st.warning(
                "This edit changed the face structure substantially. Compare it against "
                "the original carefully before reading anything into it."
            )
with detail_col:
    applied = (
        "<br>".join(
            f"&bull; {a.label} at {a.intensity}% ({a.engine}, {a.elapsed_ms:.0f} ms)"
            for a in preview.applied
        )
        or "&bull; Nothing applied yet - move a slider."
    )
    st.markdown(
        f'<div class="panel"><h4>What was applied</h4><p>{applied}</p>'
        f'<p style="margin-top:.6rem">{IDENTITY_EXPLANATION}</p></div>',
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
    if st.button("Write the notes", type="secondary"):
        with st.spinner("Writing..."):
            st.session_state["narrative"] = summarise(summary, SETTINGS)
    narrative = st.session_state.get("narrative")
    if narrative is not None:
        st.markdown(narrative.text)
        st.caption(
            f"Source: {narrative.source}" + (f" - {narrative.detail}" if narrative.detail else "")
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
    f'<div class="footer">{DISCLAIMER} Nothing here is medical advice, and no image '
    "leaves this machine: detection, editing and scoring all run locally on CPU.</div>",
    unsafe_allow_html=True,
)
