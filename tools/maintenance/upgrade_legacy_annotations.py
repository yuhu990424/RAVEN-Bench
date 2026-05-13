from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import copy
import hashlib
import json
from pathlib import Path


ROOT = ROOT_DIR
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"

PROTECTED_FILES = {
    "5th.Wheel.json",
    "Air_Canada_Airliner.json",
    "Airplane.json",
    "Automobile.json",
    "Cargo.Ship_Horizon.json",
    "Cargo.Truck.2.json",
    "Cargo.Truck.json",
}

LABELS = ["A", "B", "C", "D"]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def normalized_scenario_type(value: str | None) -> str:
    if value == "EO_IR_required":
        return "EO+IR-required"
    return value or "EO+IR-required"


def infer_h1_focus(main_event: str, description: str) -> str:
    text = f"{main_event} {description}".lower()
    if any(token in text for token in ["thermal", "solar", "exhaust", "engine", "hotspot", "heat"]):
        return "thermal_state_disambiguation"
    if any(token in text for token in ["occlusion", "hidden", "foliage", "disappear"]):
        return "cross_modal_occlusion_persistence"
    if any(token in text for token in ["stationary", "docked", "moored", "parked", "inactive"]):
        return "operational_state_reasoning"
    if any(token in text for token in ["trailer", "articulated", "coupled"]):
        return "entity_integrity_reasoning"
    if any(token in text for token in ["scale", "zoom", "tracking", "continuity", "persist"]):
        return "cross_modal_target_persistence"
    return "cross_modal_evidence_integration"


def first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "The same target-event interpretation remains supported across the EO and IR observations."
    if "." in text:
        text = text.split(".", 1)[0].strip()
    return text.rstrip(".")


def normalize_question_types(values, fallback: list[str]) -> list[str]:
    if not values:
        return fallback
    if isinstance(values, str):
        values = [values]
    mapping = {
        "entity_recognition": "entity_grounding",
        "key_information_retrieval": "evidence_retrieval",
        "cross_modal_comparison": "cross_modal_phenomenon_explanation",
        "reasoning": "causal_reasoning",
        "kinematic_analysis": "camera_motion_vs_target_motion",
    }
    return [mapping.get(v, v) for v in values]


def split_time_reference(question: dict) -> tuple[str, str]:
    eo = question.get("time_reference_eo", "")
    ir = question.get("time_reference_ir", "")
    if eo or ir:
        return eo, ir
    ref = question.get("time_reference", "")
    modality = question.get("modality_requirement", "EO+IR")
    if modality == "EO":
        return ref, ""
    if modality == "IR":
        return "", ref
    return ref, ref


def combined_time_reference(questions: list[dict], key: str) -> str:
    values: list[str] = []
    for question in questions:
        value = question.get(key, "")
        if value and value not in values:
            values.append(value)
    return "; ".join(values)


def bridge_question_payload(focus: str) -> tuple[str, dict[str, str], dict[str, str], list[str]]:
    question = "Before drawing the final event-level conclusion, what intermediate cross-modal claim is best supported by the evidence gathered so far?"
    if focus == "thermal_state_disambiguation":
        correct = "The EO appearance and the IR thermal pattern emphasize different physical cues of the same target, so both should be integrated to infer its operational state."
    elif focus in {"cross_modal_occlusion_persistence", "cross_modal_target_persistence", "entity_integrity_reasoning"}:
        correct = "The EO and IR streams provide complementary continuity evidence for the same target, even when one modality becomes less informative or more ambiguous."
    elif focus == "operational_state_reasoning":
        correct = "The visual context and the thermal evidence remain jointly consistent with one operational-state interpretation rather than with contradictory target states."
    else:
        correct = "The EO and IR views emphasize different but complementary evidence about the same target-event, so the modalities should be integrated rather than treated as contradictory."
    options = {
        "A": correct,
        "B": "IR alone overrides the EO evidence, so the visible appearance should be discarded when the modalities differ in emphasis.",
        "C": "EO alone is sufficient and the IR evidence is best treated as noise, regardless of localized thermal cues or visibility changes.",
        "D": "Any appearance difference between EO and IR implies a target switch or temporal misalignment rather than two views of the same event.",
    }
    option_roles = {
        "A": "correct",
        "B": "single_modality_override_error",
        "C": "thermal_evidence_omission_error",
        "D": "false_cross_modal_conflict",
    }
    question_types = ["cross_modal_phenomenon_explanation", "causal_reasoning"]
    return question, options, option_roles, question_types


def make_c1_group_verdict(annotation: dict, c1_questions: list[dict]) -> dict:
    desc = first_sentence(annotation.get("event_description", ""))
    correct = f"{desc}, and this same target-event interpretation remains the best-supported reading across the observed EO and IR evidence."
    return {
        "uid": "4",
        "group_id": "C1",
        "group_family": "consistency",
        "group_focus": "multi_scale_grounding",
        "group_step": 4,
        "capability_level": "L3",
        "question": "Considering the EO identification, the temporal continuity cues, and the IR observations, which global interpretation is best supported?",
        "options": {
            "A": correct,
            "B": "The later views are better explained by a different but similar-looking target replacing the original one during the sequence.",
            "C": "The target's motion or operational state changes too drastically to support one stable interpretation across the clip.",
            "D": "The EO and IR observations are better explained by a fundamental cross-modal conflict than by a single target-event interpretation.",
        },
        "answer": "A",
        "option_roles": {
            "A": "correct",
            "B": "identity_switch_error",
            "C": "state_instability_overclaim",
            "D": "false_cross_modal_conflict",
        },
        "question_type": ["group_verdict"],
        "time_reference_eo": combined_time_reference(c1_questions, "time_reference_eo"),
        "time_reference_ir": combined_time_reference(c1_questions, "time_reference_ir"),
        "modality_requirement": "EO+IR",
    }


def normalize_base_question(
    question: dict,
    *,
    group_family: str,
    group_focus: str,
    group_step: int,
    capability_level: str,
    default_types: list[str],
) -> dict:
    normalized = copy.deepcopy(question)
    normalized["group_family"] = group_family
    normalized["group_focus"] = group_focus
    normalized["group_step"] = group_step
    normalized["capability_level"] = capability_level
    normalized["question_type"] = normalize_question_types(normalized.get("question_type"), default_types)
    eo_ref, ir_ref = split_time_reference(normalized)
    normalized["time_reference_eo"] = eo_ref
    normalized["time_reference_ir"] = ir_ref
    normalized.pop("time_reference", None)
    normalized.pop("group_type", None)
    normalized.setdefault("option_roles", {})
    return normalized


def shuffle_question(annotation_name: str, question: dict) -> dict:
    digest = hashlib.sha256(f"{annotation_name}:{question['uid']}".encode()).digest()
    order = list(range(4))
    for idx in range(3, 0, -1):
        swap_idx = digest[idx] % (idx + 1)
        order[idx], order[swap_idx] = order[swap_idx], order[idx]
    old_labels = LABELS[:]
    new_options = {}
    new_roles = {}
    answer = question["answer"]
    for new_idx, old_idx in enumerate(order):
        new_label = LABELS[new_idx]
        old_label = old_labels[old_idx]
        new_options[new_label] = question["options"][old_label]
        if "option_roles" in question:
            new_roles[new_label] = question["option_roles"].get(old_label, "")
        if old_label == answer:
            new_answer = new_label
    question["options"] = new_options
    question["option_roles"] = new_roles
    question["answer"] = new_answer
    return question


def upgrade_six_question_annotation(annotation: dict, path: Path) -> dict:
    qa = annotation["qa"]
    c1_old = qa[:3]
    h1_old = qa[3:]
    h1_focus = infer_h1_focus(annotation.get("main_event", ""), annotation.get("event_description", ""))

    c1_specs = [
        ("L1", ["entity_grounding"]),
        ("L2", ["temporal_reasoning"]),
        ("L2", ["thermal_evidence_interpretation"]),
    ]
    h1_specs = [
        ("L1", ["evidence_retrieval"]),
        ("L2", ["thermal_evidence_interpretation"]),
        ("L3", ["group_verdict"]),
    ]

    c1_questions = []
    for idx, (question, (level, default_types)) in enumerate(zip(c1_old, c1_specs), start=1):
        c1_questions.append(
            normalize_base_question(
                question,
                group_family="consistency",
                group_focus="multi_scale_grounding",
                group_step=idx,
                capability_level=level,
                default_types=default_types,
            )
        )
        c1_questions[-1]["uid"] = str(idx)
    c1_questions.append(make_c1_group_verdict(annotation, c1_questions))

    h1_questions = []
    for idx, (question, (level, default_types)) in enumerate(zip(h1_old[:2], h1_specs[:2]), start=1):
        h1_questions.append(
            normalize_base_question(
                question,
                group_family="coherence",
                group_focus=h1_focus,
                group_step=idx,
                capability_level=level,
                default_types=default_types,
            )
        )
        h1_questions[-1]["uid"] = str(4 + idx)

    bridge_q, bridge_options, bridge_roles, bridge_types = bridge_question_payload(h1_focus)
    h1_questions.append(
        {
            "uid": "7",
            "group_id": "H1",
            "group_family": "coherence",
            "group_focus": h1_focus,
            "group_step": 3,
            "capability_level": "L2",
            "question": bridge_q,
            "options": bridge_options,
            "answer": "A",
            "option_roles": bridge_roles,
            "question_type": bridge_types,
            "time_reference_eo": combined_time_reference(h1_questions, "time_reference_eo"),
            "time_reference_ir": combined_time_reference(h1_questions, "time_reference_ir"),
            "modality_requirement": "EO+IR",
        }
    )

    final_question = normalize_base_question(
        h1_old[2],
        group_family="coherence",
        group_focus=h1_focus,
        group_step=4,
        capability_level="L3",
        default_types=["group_verdict"],
    )
    final_question["uid"] = "8"
    final_question["question_type"] = ["group_verdict"]
    h1_questions.append(final_question)

    annotation["qa"] = c1_questions + h1_questions
    return annotation


def normalize_eight_question_annotation(annotation: dict, path: Path) -> dict:
    h1_focus = infer_h1_focus(annotation.get("main_event", ""), annotation.get("event_description", ""))
    for question in annotation["qa"]:
        group_id = question["group_id"]
        step = int(question["group_step"])
        if group_id == "C1":
            level = "L1" if step == 1 else "L2" if step in {2, 3} else "L3"
            normalized = normalize_base_question(
                question,
                group_family="consistency",
                group_focus="multi_scale_grounding",
                group_step=step,
                capability_level=level,
                default_types=["group_verdict"] if step == 4 else ["evidence_retrieval"],
            )
        else:
            level = "L1" if step == 1 else "L2" if step in {2, 3} else "L3"
            default_types = ["group_verdict"] if step == 4 else ["evidence_retrieval"]
            normalized = normalize_base_question(
                question,
                group_family="coherence",
                group_focus=h1_focus,
                group_step=step,
                capability_level=level,
                default_types=default_types,
            )
            if step == 4:
                normalized["question_type"] = ["group_verdict"]
        question.clear()
        question.update(normalized)
    return annotation


def normalize_notes(annotation: dict, h1_focus: str) -> dict:
    annotation["notes"] = {
        "group_scoring_recommendation": {
            "C1": "Use step accuracy or consistency scoring. Step 4 is a group-verdict question and should be scored only if Steps 1-3 are correct, or reported separately.",
            "H1": f"Prefer prefix scoring because Steps 1-4 form a strict evidence chain for cross-modal {h1_focus.replace('_', ' ')} reasoning.",
        },
        "annotation_visibility_warning": "event_description should remain hidden from model input during evaluation to avoid answer leakage.",
    }
    return annotation


def upgrade_annotation(path: Path) -> bool:
    if path.name in PROTECTED_FILES:
        return False
    annotation = load_json(path)
    annotation["sample_id"] = path.stem
    annotation["scenario_type"] = normalized_scenario_type(annotation.get("scenario_type"))
    h1_focus = infer_h1_focus(annotation.get("main_event", ""), annotation.get("event_description", ""))
    normalize_notes(annotation, h1_focus)

    question_count = len(annotation.get("qa", []))
    if question_count == 6:
        annotation = upgrade_six_question_annotation(annotation, path)
    elif question_count == 8:
        annotation = normalize_eight_question_annotation(annotation, path)
    else:
        raise ValueError(f"{path.name}: unsupported question count {question_count}")

    for question in annotation["qa"]:
        shuffle_question(path.name, question)

    dump_json(path, annotation)
    return True


def main() -> None:
    changed = 0
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        if upgrade_annotation(path):
            changed += 1
    print({"changed_files": changed})


if __name__ == "__main__":
    main()
