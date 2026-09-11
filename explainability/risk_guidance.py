# explainability/risk_guidance.py
# -----------------------------------------------------------------------------
# Human-readable explanation + recommended action for each risk tier.
# Used by api.py to answer "why is this High/Moderate/Low risk, and what
# should I do about it?" alongside the model's raw prediction.
#
# This is rule-based (keyed on the predicted risk label), not a per-input
# explanation - it explains what that risk tier means and the standard
# aquaculture response, not which exact reading or pixel drove the decision
# (that's what the SHAP chart / a future Grad-CAM overlay is for).
#
# Two separate guidance sets exist because the two model branches see
# different evidence: the sensor branch reasons over water chemistry
# (DO, pH, temperature, ammonia) while the vision branch reasons over a
# photo of the water surface. Reusing the same wording for both would
# describe evidence the sensor branch never saw.
# -----------------------------------------------------------------------------

SENSOR_GUIDANCE = {
    "Low": {
        "why": (
            "Current water-quality readings (dissolved oxygen, pH, temperature, "
            "ammonia) are within a safe range for fish, and the 12-hour oxygen "
            "forecast does not show signs of a crash."
        ),
        "precautions": [
            "Continue routine monitoring (DO, pH, temperature, turbidity, ammonia).",
            "Re-check after rainfall or a feeding change, which can shift water chemistry quickly.",
            "No emergency action needed at this time.",
        ],
    },
    "Moderate": {
        "why": (
            "One or more readings are drifting toward unsafe territory - for "
            "example dissolved oxygen trending down, ammonia elevated, or "
            "temperature running high - and the 12-hour forecast suggests "
            "oxygen could dip further overnight."
        ),
        "precautions": [
            "Increase monitoring frequency - check DO at dawn, when levels are lowest.",
            "Verify aerators are working and ready to run overnight if needed.",
            "Consider a partial water exchange if ammonia or turbidity is elevated.",
            "Reduce feeding slightly to limit additional oxygen demand from digestion.",
        ],
    },
    "High": {
        "why": (
            "Sensor readings indicate a critical condition - e.g. dissolved "
            "oxygen already low, ammonia elevated, or heat/pH stress - and the "
            "12-hour forecast points to a further oxygen crash, the leading "
            "cause of sudden fish kills."
        ),
        "precautions": [
            "Activate aerators immediately, especially before dawn.",
            "Halt feeding - digestion consumes oxygen the pond can't spare right now.",
            "Consider emergency water exchange to dilute ammonia/organic load.",
            "Re-test DO and ammonia directly to confirm sensor readings.",
            "Alert farm staff - be ready for emergency harvest or fish relocation if DO keeps dropping.",
        ],
    },
}

IMAGE_GUIDANCE = {
    "Low": {
        "why": (
            "The water surface shows little to no visible algal scum - color "
            "and clarity are close to normal, indicating cyanobacteria/"
            "chlorophyll concentration is likely low right now."
        ),
        "precautions": [
            "Continue routine visual checks of the pond surface.",
            "Re-check after rainfall or nutrient runoff, which can trigger new blooms quickly.",
            "No emergency action needed at this time.",
        ],
    },
    "Moderate": {
        "why": (
            "Visible green discoloration or patchy algal mats are present in "
            "the photo, consistent with an active but not yet severe bloom. "
            "This can begin causing larger day/night swings in dissolved oxygen."
        ),
        "precautions": [
            "Increase monitoring frequency - check DO at dawn, when levels are lowest.",
            "Verify aerators are working and ready to run overnight if needed.",
            "Consider a partial water exchange to dilute nutrient load.",
            "Reduce feeding slightly to limit additional oxygen demand from digestion.",
        ],
    },
    "High": {
        "why": (
            "Dense, widespread algal mat/scum covers most of the visible "
            "surface in the photo - consistent with a severe bloom. Heavy "
            "blooms cause large overnight oxygen crashes as algae respire and "
            "decompose, the leading cause of sudden fish kills."
        ),
        "precautions": [
            "Activate aerators immediately, especially before dawn.",
            "Halt feeding - digestion consumes oxygen the pond can't spare right now.",
            "Consider emergency water exchange to dilute the bloom.",
            "Apply an approved algaecide/bio-cleaner per local regulations, if available.",
            "Alert farm staff - be ready for emergency harvest or fish relocation if DO keeps dropping.",
        ],
    },
}


def get_guidance(risk_label: str, source: str = "sensor") -> dict:
    """
    Returns {"why": str, "precautions": [str, ...]} for a risk label.

    source: "sensor" (BiLSTM + tabular classifier) or "image" (CNN vision branch) -
    each branch sees different evidence, so each gets its own explanation text.
    """
    guidance_set = IMAGE_GUIDANCE if source == "image" else SENSOR_GUIDANCE
    return guidance_set.get(risk_label, guidance_set["Moderate"])
