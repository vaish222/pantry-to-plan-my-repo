"""Independent Streamlit UI for image detection and pantry confirmation.

Run from the repository root:
    streamlit run vision_app.py

This application stops at PantryState. It does not import or invoke planning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from normalization import IngredientNormalizer
from schemas import PantryItemCandidate
from vision import (
    CompatibleVisionProvider,
    MockVisionProvider,
    VisionPipelineError,
    parse_pantry_image,
)
from llm_provider import provider_chain
from vision_ui_helpers import candidates_to_editor_rows, editor_rows_to_pantry


EDITOR_COLUMNS = [
    "include",
    "display_name",
    "ingredient_id",
    "quantity_g",
    "confidence",
    "source_text",
    "normalization_status",
    "recipe_supported",
]


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

st.set_page_config(page_title="Pantry Vision", page_icon="🥫", layout="wide")
st.title("Pantry Vision")
st.caption("Upload → detect → review → confirm. This UI stops before meal planning.")

with st.sidebar:
    st.header("Detection settings")
    mode = st.radio("Provider", ("Live (Ollama → Nebius)", "Mock demo"), index=0)
    threshold = st.slider("Low-confidence threshold", 0.0, 1.0, 0.75, 0.05)
    if mode.startswith("Live"):
        configured = provider_chain("vision")
        st.caption(" → ".join(f"{item.name}: `{item.model}`" for item in configured))
        if not configured:
            st.warning("Start Ollama or configure NEBIUS_API_KEY in .env.")
    st.info("Quantities stay blank unless an exact gram value is readable.")

uploaded = st.file_uploader(
    "Upload a pantry photo",
    type=("jpg", "jpeg", "png", "webp"),
    help="JPEG, PNG or WebP; maximum 10 MB.",
)

if uploaded is not None:
    left, right = st.columns((1, 1))
    with left:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
    with right:
        st.subheader("Before analysis")
        st.write(
            "The model proposes visible ingredients. You remain responsible for "
            "correcting, removing and adding items before confirmation."
        )
        analyze = st.button("Analyze pantry", type="primary", use_container_width=True)

    if analyze:
        try:
            provider = MockVisionProvider() if mode == "Mock demo" else CompatibleVisionProvider()
            with st.spinner("Detecting and normalizing ingredients..."):
                result = parse_pantry_image(
                    uploaded.getvalue(),
                    provider=provider,
                    normalizer=IngredientNormalizer(),
                    low_confidence_threshold=threshold,
                )
        except VisionPipelineError as exc:
            st.error(exc.issue.message)
            if exc.issue.suggested_action:
                st.info(exc.issue.suggested_action)
        except Exception as exc:
            st.error(f"Could not start detection: {exc}")
        else:
            st.session_state.vision_result = result.model_dump(mode="json")
            st.session_state.vision_editor_rows = candidates_to_editor_rows(result.items)
            st.session_state.vision_editor_version = (
                st.session_state.get("vision_editor_version", 0) + 1
            )
            st.session_state.pop("confirmed_pantry_json", None)

if "vision_result" in st.session_state:
    result_data = st.session_state.vision_result
    st.divider()
    st.subheader("Review detected ingredients")
    metadata_left, metadata_right = st.columns(2)
    metadata_left.metric("Provider", result_data["provider"])
    metadata_right.metric("Model latency", f'{result_data["model_latency_ms"]} ms')

    issues = result_data.get("issues", [])
    if issues:
        with st.expander(f"Review notes ({len(issues)})", expanded=True):
            for issue in issues:
                st.warning(f'**{issue["code"]}** — {issue["message"]}')

    editor_frame = pd.DataFrame(
        st.session_state.vision_editor_rows,
        columns=EDITOR_COLUMNS,
    )
    edited_frame = st.data_editor(
        editor_frame,
        key=f'vision_editor_{st.session_state.get("vision_editor_version", 0)}',
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "include": st.column_config.CheckboxColumn("Include", default=True),
            "display_name": st.column_config.TextColumn("Ingredient", required=True),
            "ingredient_id": st.column_config.TextColumn(
                "Canonical ID", required=True, help="Lowercase snake_case."
            ),
            "quantity_g": st.column_config.NumberColumn(
                "Quantity (g)", min_value=0.0, step=1.0
            ),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "source_text": st.column_config.TextColumn("Detected label"),
            "normalization_status": st.column_config.TextColumn("Mapping"),
            "recipe_supported": st.column_config.CheckboxColumn("Used by recipes"),
        },
        disabled=("confidence", "source_text", "normalization_status", "recipe_supported"),
    )

    st.caption(
        "You may edit ingredient names and canonical IDs, remove rows, or clear "
        "Include. A canonical ID must use lowercase snake_case."
    )

    with st.expander("Add an ingredient manually"):
        manual_name = st.text_input("Ingredient name", key="manual_ingredient_name")
        if st.button("Add ingredient"):
            if not manual_name.strip():
                st.warning("Enter an ingredient name first.")
            else:
                normalized = IngredientNormalizer().normalize(manual_name)
                candidate = PantryItemCandidate(
                    ingredient_id=normalized.ingredient_id,
                    display_name=normalized.display_name,
                    quantity_g=None,
                    confidence=1.0,
                    source_text="user_added",
                    normalization_status=normalized.status,
                    recipe_supported=normalized.recipe_supported,
                )
                st.session_state.vision_editor_rows.append(
                    candidates_to_editor_rows([candidate])[0]
                )
                st.session_state.vision_editor_version = (
                    st.session_state.get("vision_editor_version", 0) + 1
                )
                st.rerun()

    if st.button("Confirm pantry", type="primary"):
        try:
            pantry = editor_rows_to_pantry(edited_frame.to_dict(orient="records"))
        except (ValidationError, ValueError, TypeError) as exc:
            st.error(f"Please correct the table before confirming: {exc}")
        else:
            st.session_state.confirmed_pantry_json = pantry.model_dump_json(indent=2)

if "confirmed_pantry_json" in st.session_state:
    st.divider()
    st.success("Pantry confirmed. It is ready for the planner handoff.")
    st.code(st.session_state.confirmed_pantry_json, language="json")
    st.download_button(
        "Download confirmed_pantry.json",
        data=st.session_state.confirmed_pantry_json,
        file_name="confirmed_pantry.json",
        mime="application/json",
        type="primary",
    )
