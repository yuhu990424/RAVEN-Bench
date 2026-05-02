from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / "annotations"
PROTECTED_FILES = {
    "5th.Wheel.json",
    "Air_Canada_Airliner.json",
    "Airplane.json",
    "Automobile.json",
    "Cargo.Ship_Horizon.json",
    "Cargo.Truck.2.json",
    "Cargo.Truck.json",
    "Container.Ship.json",
    "Covered.Boat.json",
    "Docked Ferry.2.json",
}
LABELS = ["A", "B", "C", "D"]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "The same target-event interpretation remains supported across the EO and IR evidence."
    if "." in text:
        text = text.split(".", 1)[0].strip()
    return text.rstrip(".")


def lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def strip_qmark(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:-1] if text.endswith("?") else text


def infer_motion_profile(text: str) -> str:
    lowered = text.lower()
    stationary_terms = [
        "stationary",
        "docked",
        "moored",
        "parked",
        "anchored",
        "sitting",
        "inactive",
        "remains on the ground",
    ]
    moving_terms = [
        "moving",
        "driving",
        "traveling",
        "tracked",
        "navigating",
        "under way",
        "underway",
        "continues along",
        "crossing",
    ]
    if any(token in lowered for token in stationary_terms):
        return "stationary"
    if any(token in lowered for token in moving_terms):
        return "moving"
    return "ambiguous"


def infer_focus(annotation: dict) -> str:
    text = f"{annotation.get('main_event', '')} {annotation.get('event_description', '')}".lower()
    if any(token in text for token in ["thermal", "solar", "heat", "hotspot", "exhaust", "engine", "warming"]):
        return "thermal_state_disambiguation"
    if any(token in text for token in ["occlusion", "hidden", "foliage", "disappear"]):
        return "cross_modal_occlusion_persistence"
    if any(token in text for token in ["wake", "underway", "moored", "docked", "parked", "stationary"]):
        return "operational_state_reasoning"
    if any(token in text for token in ["trailer", "articulated", "coupled", "separation"]):
        return "entity_integrity_reasoning"
    if any(token in text for token in ["trajectory", "scale", "zoom", "tracking", "continuity"]):
        return "cross_modal_target_persistence"
    return "cross_modal_evidence_integration"


def time_ref(question: dict, key: str) -> str:
    return question.get(key, "")


def combined_refs(questions: list[dict], key: str) -> str:
    values: list[str] = []
    for question in questions:
        value = time_ref(question, key)
        if value and value not in values:
            values.append(value)
    return "; ".join(values)


def subtle_q4(annotation: dict, qa: list[dict]) -> dict:
    summary = first_sentence(annotation.get("event_description", ""))
    motion = infer_motion_profile(summary)
    if motion == "stationary":
        question = (
            "After reconciling the EO identity cues, the temporal evidence, and the IR observations, "
            "which interpretation is most defensible about the target's identity and state?"
        )
        correct = (
            f"{summary}, and the combined EO/IR evidence still supports one stable target-state interpretation "
            "rather than departure, replacement, or a late change in operational state."
        )
        distractors = {
            "B": "The target category may be right, but the later observations are too weak to rule out a late departure or state transition.",
            "C": "The later views are better explained by a different but visually similar target replacing the original one during the sequence.",
            "D": "The EO and IR evidence emphasize sufficiently different states that no single target-level interpretation should be preferred.",
        }
        roles = {
            "A": "correct",
            "B": "state_transition_overclaim",
            "C": "identity_switch_error",
            "D": "false_cross_modal_conflict",
        }
    elif motion == "moving":
        question = (
            "After reconciling the EO structure, the temporal continuity cues, and the IR observations, "
            "which interpretation is best supported for the tracked event?"
        )
        correct = (
            f"{summary}, and the joint EO/IR evidence supports continuity of the same target through "
            "scale or viewpoint changes rather than stop, reversal, or substitution."
        )
        distractors = {
            "B": "The target class is plausible, but continuity across the later views is too weak to support one stable identity or motion-state interpretation.",
            "C": "The later observations are more consistent with a stop, turnaround, or track break than with a single target continuing through the scene.",
            "D": "The EO and IR streams support different target states strongly enough that one unified event interpretation is not justified.",
        }
        roles = {
            "A": "correct",
            "B": "continuity_underclaim_error",
            "C": "trajectory_discontinuity_hallucination",
            "D": "false_cross_modal_conflict",
        }
    else:
        question = (
            "After reconciling the EO evidence, the IR evidence, and the temporal context, "
            "which interpretation remains the most defensible overall?"
        )
        correct = (
            f"{summary}, and the modalities remain complementary enough to support one coherent target-event interpretation."
        )
        distractors = {
            "B": "The general category is clear, but the clip is too ambiguous to support a stable event-level interpretation.",
            "C": "Later observations are better explained by a different but similar target than by one continuous event.",
            "D": "The EO and IR evidence conflict too strongly for any single interpretation to remain credible.",
        }
        roles = {
            "A": "correct",
            "B": "evidence_underclaim_error",
            "C": "identity_switch_error",
            "D": "false_cross_modal_conflict",
        }
    return {
        "uid": "4",
        "group_id": "C1",
        "group_family": "consistency",
        "group_focus": "multi_scale_grounding",
        "group_step": 4,
        "capability_level": "L3",
        "question": question,
        "options": {"A": correct, **distractors},
        "answer": "A",
        "option_roles": roles,
        "question_type": ["group_verdict"],
        "time_reference_eo": combined_refs(qa[:3], "time_reference_eo"),
        "time_reference_ir": combined_refs(qa[:3], "time_reference_ir"),
        "modality_requirement": "EO+IR",
    }


def subtle_q7(annotation: dict, qa: list[dict]) -> dict:
    focus = infer_focus(annotation)
    summary = first_sentence(annotation.get("event_description", ""))
    motion = infer_motion_profile(summary)
    eo_ref = combined_refs(qa[4:6], "time_reference_eo")
    ir_ref = combined_refs(qa[4:6], "time_reference_ir")

    if focus == "thermal_state_disambiguation":
        question = (
            "Before accepting the final event-level conclusion, which intermediate cross-modal interpretation "
            "is necessary to keep the thermal evidence physically grounded?"
        )
        correct = (
            "Localized or broad IR brightness has to be interpreted together with the EO structural and motion cues, "
            "so environmental heating, auxiliary activity, and true propulsion evidence are not conflated."
        )
        distractors = {
            "B": "Any clear hotspot is enough to conclude full-power propulsion or immediate departure, even when the EO evidence does not support that escalation.",
            "C": "Because EO and IR emphasize different structures, the safest interpretation is that they depict mismatched targets or different times.",
            "D": "The EO view alone should determine the conclusion, because the IR evidence is too ambiguous to contribute meaningful state information.",
        }
        roles = {
            "A": "correct",
            "B": "causal_overreach_error",
            "C": "false_cross_modal_conflict",
            "D": "thermal_evidence_omission_error",
        }
    elif focus in {"cross_modal_target_persistence", "cross_modal_occlusion_persistence", "entity_integrity_reasoning"}:
        question = (
            "Before accepting the final conclusion, which intermediate continuity claim is best supported "
            "once the EO and IR streams are considered together?"
        )
        correct = (
            "Cross-modal continuity should be preserved unless the evidence supports a real break in target identity, "
            "so brief ambiguity, clutter, or modality-specific salience changes should not be over-read as separation or substitution."
        )
        distractors = {
            "B": "Whenever one modality becomes less descriptive, the safer interpretation is that the original target has likely been replaced by a similar one.",
            "C": "Any temporary offset or visibility loss is stronger evidence for a stop, split, or reversal than for continuity of the same target.",
            "D": "The IR evidence should dominate the interpretation even if it weakens the temporal continuity already established in EO.",
        }
        roles = {
            "A": "correct",
            "B": "identity_switch_error",
            "C": "trajectory_break_hallucination",
            "D": "single_modality_override_error",
        }
    else:
        question = (
            "Before accepting the final event-level conclusion, which intermediate multimodal claim is best supported?"
        )
        correct = (
            "The EO context and the IR evidence constrain the same event interpretation, so neither modality should be treated as decisive in isolation."
        )
        distractors = {
            "B": "Any difference in salience between EO and IR is stronger evidence for contradiction than for complementary sensing.",
            "C": "The IR evidence should override the EO evidence whenever it appears more diagnostic, even if that breaks temporal consistency.",
            "D": "The EO evidence alone is sufficient, and the IR view should only be used as a weak sanity check rather than part of the reasoning chain.",
        }
        roles = {
            "A": "correct",
            "B": "false_cross_modal_conflict",
            "C": "single_modality_override_error",
            "D": "modality_underuse_error",
        }

    return {
        "uid": "7",
        "group_id": "H1",
        "group_family": "coherence",
        "group_focus": focus,
        "group_step": 3,
        "capability_level": "L3" if focus in {"thermal_state_disambiguation", "operational_state_reasoning"} else "L2",
        "question": question,
        "options": {"A": correct, **distractors},
        "answer": "A",
        "option_roles": roles,
        "question_type": ["cross_modal_phenomenon_explanation", "causal_reasoning"],
        "time_reference_eo": eo_ref,
        "time_reference_ir": ir_ref,
        "modality_requirement": "EO+IR",
    }


def subtle_q8(annotation: dict, qa: list[dict]) -> dict:
    focus = infer_focus(annotation)
    summary = first_sentence(annotation.get("event_description", ""))
    motion = infer_motion_profile(summary)
    question = (
        "When the EO structure, the IR evidence, and the full temporal context are considered together, "
        "which final conclusion is most defensible?"
    )
    if focus == "thermal_state_disambiguation":
        correct = (
            f"{summary}, and the thermal evidence is better explained by a constrained, physically grounded state interpretation "
            "than by an unsupported escalation such as full propulsion, emergency failure, or modality conflict."
        )
        distractors = {
            "B": "The broad target identity is clear, but the thermal evidence is too unstable to support any specific operational-state conclusion.",
            "C": "The IR evidence is strong enough to override the EO motion cues and imply a substantially different target state than the visual sequence suggests.",
            "D": "The EO and IR views are best read as contradictory observations of different target conditions rather than a single event.",
        }
        roles = {
            "A": "correct",
            "B": "evidence_dilution_error",
            "C": "single_modality_override_error",
            "D": "false_cross_modal_conflict",
        }
    elif motion == "stationary":
        correct = (
            f"{summary}, and the joint EO/IR evidence still favors one stable stationary or dockside interpretation "
            "over explanations based on departure, replacement, or unsupported state change."
        )
        distractors = {
            "B": "The target can be identified, but the clip does not support a stronger claim than generic category recognition once both modalities are considered.",
            "C": "The later observations are more consistent with the target transitioning into active movement or imminent departure than with staying in its earlier state.",
            "D": "The modalities remain too inconsistent to support one coherent event-level conclusion.",
        }
        roles = {
            "A": "correct",
            "B": "group_underclaim_error",
            "C": "unsupported_state_change",
            "D": "false_cross_modal_conflict",
        }
    else:
        correct = (
            f"{summary}, and the combined EO/IR evidence supports continuity of the same event more strongly "
            "than explanations based on track break, target substitution, or unsupported reversal."
        )
        distractors = {
            "B": "The general target type is clear, but once all evidence is considered the clip remains too ambiguous for a single event-level conclusion.",
            "C": "The later observations are better interpreted as a stop, split, or target replacement than as continuity of the same event.",
            "D": "The EO and IR streams provide incompatible readings of the event, so a unified conclusion is not defensible.",
        }
        roles = {
            "A": "correct",
            "B": "group_underclaim_error",
            "C": "identity_or_trajectory_break_error",
            "D": "false_cross_modal_conflict",
        }
    return {
        "uid": "8",
        "group_id": "H1",
        "group_family": "coherence",
        "group_focus": focus,
        "group_step": 4,
        "capability_level": "L3",
        "question": question,
        "options": {"A": correct, **distractors},
        "answer": "A",
        "option_roles": roles,
        "question_type": ["group_verdict"],
        "time_reference_eo": combined_refs(qa[4:8], "time_reference_eo"),
        "time_reference_ir": combined_refs(qa[4:8], "time_reference_ir"),
        "modality_requirement": "EO+IR",
    }


def rebalance_stem(question: dict) -> None:
    uid = question["uid"]
    original = strip_qmark(question["question"])
    prefixes = {
        "1": "Using only the earliest referenced evidence and avoiding over-commitment to fine detail,",
        "2": "When separating target behavior from camera, viewpoint, or scale-change effects,",
        "3": "Focusing on the referenced cross-modal or thermal evidence rather than on raw salience alone,",
        "5": "From the referenced evidence window alone, and without importing assumptions from later frames,",
        "6": "Interpreting the referenced IR evidence conservatively and in a physically grounded way,",
    }
    prefix = prefixes.get(uid)
    if not prefix:
        return
    question["question"] = f"{prefix} {lower_first(original)}?"


def shuffle_question(path: Path, question: dict) -> None:
    digest = hashlib.sha256(f"{path.name}:{question['uid']}:hard_v2".encode()).digest()
    order = list(range(4))
    for idx in range(3, 0, -1):
        swap_idx = digest[idx] % (idx + 1)
        order[idx], order[swap_idx] = order[swap_idx], order[idx]
    old_options = question["options"]
    old_roles = question["option_roles"]
    old_answer = question["answer"]
    new_options = {}
    new_roles = {}
    new_answer = None
    for new_idx, old_idx in enumerate(order):
        new_label = LABELS[new_idx]
        old_label = LABELS[old_idx]
        new_options[new_label] = old_options[old_label]
        new_roles[new_label] = old_roles[old_label]
        if old_label == old_answer:
            new_answer = new_label
    question["options"] = new_options
    question["option_roles"] = new_roles
    question["answer"] = new_answer


def harden_annotation(path: Path) -> bool:
    if path.name in PROTECTED_FILES:
        return False
    annotation = load_json(path)
    qa = annotation["qa"]
    for idx in [0, 1, 2, 4, 5]:
        rebalance_stem(qa[idx])
    qa[3] = subtle_q4(annotation, qa)
    qa[6] = subtle_q7(annotation, qa)
    qa[7] = subtle_q8(annotation, qa)
    for idx in [3, 6, 7]:
        shuffle_question(path, qa[idx])
    dump_json(path, annotation)
    return True


def main() -> None:
    changed = 0
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        if harden_annotation(path):
            changed += 1
    print({"changed_files": changed})


if __name__ == "__main__":
    main()
