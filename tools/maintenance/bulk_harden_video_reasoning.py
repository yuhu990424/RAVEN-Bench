from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hashlib
import json
from pathlib import Path


ROOT = ROOT_DIR
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
LABELS = ["A", "B", "C", "D"]
ALREADY_HARDENED = {
    "5th.Wheel.json",
    "Air_Canada_Airliner.json",
    "Airplane.json",
    "Automobile.json",
    "Cargo.Ship_Horizon.json",
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def first_sentence(text: str) -> str:
    text = normalize(text)
    if not text:
        return "The same target-event interpretation remains supported across the EO and IR evidence."
    if "." in text:
        text = text.split(".", 1)[0].strip()
    return text.rstrip(".")


def infer_motion_profile(annotation: dict) -> str:
    text = f"{annotation.get('main_event', '')} {annotation.get('event_description', '')}".lower()
    stationary_terms = [
        "stationary",
        "docked",
        "moored",
        "parked",
        "anchored",
        "inactive",
        "on stand",
    ]
    moving_terms = [
        "moving",
        "driving",
        "traveling",
        "tracked",
        "navigating",
        "under way",
        "underway",
        "continues",
        "progression",
        "transit",
        "motion",
        "route",
        "highway",
        "roadway",
    ]
    if any(token in text for token in stationary_terms):
        return "stationary"
    if any(token in text for token in moving_terms):
        return "moving"
    return "ambiguous"


def infer_domain(annotation: dict) -> str:
    text = f"{annotation.get('type', '')} {annotation.get('main_event', '')} {annotation.get('event_description', '')}".lower()
    if any(token in text for token in ["ship", "boat", "ferry", "vessel", "barge", "tanker", "yacht"]):
        return "maritime"
    if any(token in text for token in ["aircraft", "airplane", "airliner", "helicopter", "runway", "airstrip"]):
        return "aircraft"
    if any(token in text for token in ["truck", "van", "bus", "car", "automobile", "suv", "road", "highway"]):
        return "road"
    return "generic"


def infer_focus(annotation: dict, question: dict) -> str:
    if question.get("group_focus"):
        return question["group_focus"]
    text = f"{annotation.get('main_event', '')} {annotation.get('event_description', '')}".lower()
    if any(token in text for token in ["thermal", "solar", "heat", "hotspot", "exhaust", "engine", "warming"]):
        return "thermal_state_disambiguation"
    if any(token in text for token in ["occlusion", "hidden", "foliage", "disappear"]):
        return "cross_modal_occlusion_persistence"
    if any(token in text for token in ["trailer", "articulated", "coupled", "separation"]):
        return "truck_trailer_identity_persistence"
    if any(token in text for token in ["route", "corridor", "roadway", "landmark"]):
        return "cross_modal_route_progression"
    return "multi_scale_grounding"


def relabel_question(path: Path, question: dict, salt: str) -> None:
    old_options = question["options"]
    old_roles = question["option_roles"]
    old_answer = question["answer"]
    digest = hashlib.sha256(f"{path.name}:{question['uid']}:{salt}".encode()).digest()
    order = list(range(4))
    for idx in range(3, 0, -1):
        swap_idx = digest[idx] % (idx + 1)
        order[idx], order[swap_idx] = order[swap_idx], order[idx]
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


def rewrite_q2(annotation: dict, question: dict) -> None:
    motion = infer_motion_profile(annotation)
    domain = infer_domain(annotation)
    modality = question["modality_requirement"]

    if modality == "EO":
        if motion == "stationary":
            question["question"] = (
                "When the full referenced span is judged as continuous video rather than disconnected views, "
                "which rank ordering is best supported among one fixed-place account, one slight local reposition ambiguity, "
                "and one real departure or taxi-underway account?"
            )
            question["options"] = {
                "A": "One fixed-place account leads clearly, while slight local reposition ambiguity and real departure or taxi-underway both fall well behind.",
                "B": "One fixed-place account and one slight local reposition ambiguity remain close enough that neither clearly outranks the other, while real departure or taxi-underway trails.",
                "C": "One slight local reposition ambiguity leads, because the continuous span weakens fixed-place continuity more than it weakens low-level motion alternatives.",
                "D": "One real departure or taxi-underway account leads, because changing view and scale explain the sequence less well than genuine target movement does.",
            }
            question["option_roles"] = {
                "A": "fixed_place_overclosure",
                "B": "correct",
                "C": "micro_reposition_overclaim",
                "D": "motion_hallucination",
            }
            question["answer"] = "B"
        elif motion == "moving":
            if domain == "road":
                question["question"] = (
                    "When the referenced EO span is judged as continuous video rather than disconnected glimpses, "
                    "which rank ordering is best supported among one same specific routed target, one same generic route participant, "
                    "and one hidden branch, exit, or same-corridor replacement account?"
                )
                question["options"] = {
                    "A": "One same specific routed target leads clearly, one same generic route participant is the weaker rival, and a hidden branch, exit, or same-corridor replacement account trails.",
                    "B": "One same specific routed target and one same generic route participant remain close enough that neither clearly outranks the other, while a hidden branch, exit, or same-corridor replacement account trails.",
                    "C": "One same generic route participant leads, because the video carries route continuity better than exact target identity through the changing views.",
                    "D": "One hidden branch, exit, or same-corridor replacement account leads, because the EO span preserves too little path-locked continuity to keep a carried target account in front.",
                }
                question["option_roles"] = {
                    "A": "specific_identity_overclosure",
                    "B": "correct",
                    "C": "generic_chain_overclaim",
                    "D": "route_break_bias",
                }
                question["answer"] = "B"
            elif domain == "maritime":
                question["question"] = (
                    "When the referenced EO span is judged as continuous video rather than disconnected sightings, "
                    "which rank ordering is best supported among one same-vessel continuous-transit account, one same-corridor substitute-vessel account, "
                    "and one hidden slowing or near-stationary account?"
                )
                question["options"] = {
                    "A": "One same-vessel continuous-transit account leads clearly, while same-corridor substitution and hidden slowing both fall well behind.",
                    "B": "One same-vessel continuous-transit account and one same-corridor substitute-vessel account remain close enough that neither clearly outranks the other, while hidden slowing trails.",
                    "C": "One same-corridor substitute-vessel account leads, because the long-range or scale-changing transfer stays too weak for one preserved vessel to remain in front.",
                    "D": "One hidden slowing or near-stationary account leads, because changing scale explains the apparent progression less well than intermittent or nearly static observation does.",
                }
                question["option_roles"] = {
                    "A": "transit_overclosure",
                    "B": "correct",
                    "C": "substitution_bias",
                    "D": "hidden_slowing_hallucination",
                }
                question["answer"] = "B"
            else:
                question["question"] = (
                    "When the referenced EO span is judged as continuous video rather than endpoints alone, "
                    "which rank ordering is best supported among one preserved same-target progression account, one same-corridor replacement account, "
                    "and one hidden maneuver or state-break account?"
                )
                question["options"] = {
                    "A": "One preserved same-target progression account leads clearly, while same-corridor replacement and hidden maneuver or state break both fall well behind.",
                    "B": "One preserved same-target progression account and one same-corridor replacement account remain close enough that neither clearly outranks the other, while hidden maneuver or state break trails.",
                    "C": "One same-corridor replacement account leads, because the continuous span preserves too little path-locked continuity to keep the same target ahead.",
                    "D": "One hidden maneuver or state-break account leads, because the intermediate span is too weak to penalize brief departure, stop, or reset.",
                }
                question["option_roles"] = {
                    "A": "continuity_overclosure",
                    "B": "correct",
                    "C": "replacement_bias",
                    "D": "hidden_maneuver_hallucination",
                }
                question["answer"] = "B"
        else:
            question["question"] = (
                "When the referenced EO span is judged as continuous video rather than isolated views, "
                "which rank ordering is best supported among one preserved same-target account, one downgraded generic-target account, "
                "and one replacement or state-break account?"
            )
            question["options"] = {
                "A": "One preserved same-target account leads clearly, while downgraded generic-target and replacement or state-break accounts both fall well behind.",
                "B": "One preserved same-target account and one downgraded generic-target account remain close enough that neither clearly outranks the other, while replacement or state-break trails.",
                "C": "One downgraded generic-target account leads, because the continuous span is too weak to keep a richer target account ahead.",
                "D": "One replacement or state-break account leads, because the sequence preserves too little continuity to support one carried target.",
            }
            question["option_roles"] = {
                "A": "identity_overclosure",
                "B": "correct",
                "C": "generic_demotion_bias",
                "D": "identity_reset_bias",
            }
            question["answer"] = "B"
    elif modality == "IR":
        question["question"] = (
            "When the referenced IR span is judged as a temporal sequence rather than as a single salient frame, "
            "which rank ordering is best supported among one preserved target-evolution account, one weaker downgraded continuity account, "
            "and one clutter, artifact, or reset account?"
        )
        question["options"] = {
            "A": "One preserved target-evolution account leads clearly, one weaker downgraded continuity account is the rival, and clutter, artifact, or reset trails.",
            "B": "One preserved target-evolution account and one weaker downgraded continuity account remain close enough that neither clearly outranks the other, while clutter, artifact, or reset trails.",
            "C": "One weaker downgraded continuity account leads, because the IR span carries temporal continuity better than specific target state or identity.",
            "D": "One clutter, artifact, or reset account leads, because the IR span never stabilizes enough across time to keep a carried target account in front.",
        }
        question["option_roles"] = {
            "A": "thermal_continuity_overclosure",
            "B": "correct",
            "C": "downgraded_continuity_overclaim",
            "D": "artifact_bias",
        }
        question["answer"] = "B"
    else:
        question["question"] = (
            "When the referenced EO and IR span is judged as one continuous multimodal sequence, "
            "which rank ordering is best supported among one preserved same-target account, one weaker downgraded alternative, "
            "and one replacement or true state-break account?"
        )
        question["options"] = {
            "A": "One preserved same-target account leads clearly, while the downgraded alternative and replacement or state-break accounts both fall well behind.",
            "B": "One preserved same-target account and one weaker downgraded alternative remain close enough that neither clearly outranks the other, while replacement or state-break trails.",
            "C": "One weaker downgraded alternative leads, because the EO and IR span is too under-constrained to keep a richer same-target account ahead.",
            "D": "One replacement or true state-break account leads, because the multimodal sequence never becomes coherent enough to preserve one target account.",
        }
        question["option_roles"] = {
            "A": "multimodal_overclosure",
            "B": "correct",
            "C": "underclosed_demotion_bias",
            "D": "multimodal_break_bias",
        }
        question["answer"] = "B"


def rewrite_q6(annotation: dict, question: dict) -> None:
    focus = infer_focus(annotation, question)
    motion = infer_motion_profile(annotation)
    domain = infer_domain(annotation)
    modality = question["modality_requirement"]

    if modality == "IR":
        thermal_span = "When the referenced IR evidence is judged"
        thermal_moment = "When the referenced IR evidence is judged"
        temporal_span = "When the referenced IR span is judged"
    elif modality == "EO+IR":
        thermal_span = "When the referenced EO and IR evidence is judged"
        thermal_moment = "When the referenced EO and IR evidence is judged"
        temporal_span = "When the referenced EO and IR span is judged"
    else:
        thermal_span = "When the referenced evidence is judged"
        thermal_moment = "When the referenced evidence is judged"
        temporal_span = "When the referenced span is judged"

    if focus in {"thermal_state_disambiguation", "operational_state_reasoning"}:
        if motion == "stationary":
            question["question"] = (
                f"{thermal_span} over time rather than by one bright region, "
                "which rank ordering is best supported among passive structure or warming, low-level or residual activity, "
                "and hidden strong propulsion or engine-driven heating?"
            )
            question["options"] = {
                "A": "Passive structure or warming leads clearly, low-level or residual activity is the weaker rival, and hidden strong propulsion or engine-driven heating trails.",
                "B": "Passive structure or warming and low-level or residual activity remain close enough that neither clearly outranks the other, while hidden strong propulsion or engine-driven heating trails.",
                "C": "Low-level or residual activity leads, passive structure or warming is the weaker rival, and hidden strong propulsion or engine-driven heating trails.",
                "D": "Hidden strong propulsion or engine-driven heating leads, because any organized thermal structure should be treated as operational first.",
            }
            question["option_roles"] = {
                "A": "passive_overclosure",
                "B": "correct",
                "C": "low_activity_overclaim",
                "D": "hidden_thrust_overclaim",
            }
            question["answer"] = "B"
        elif domain == "maritime":
            question["question"] = (
                f"{thermal_span} across time rather than by one hotspot alone, "
                "which rank ordering is best supported among active self-propulsion, assisted motion, and passive drift?"
            )
            question["options"] = {
                "A": "Active self-propulsion leads clearly, assisted motion is the weaker rival, and passive drift trails.",
                "B": "Active self-propulsion and assisted motion remain close enough that neither clearly outranks the other, while passive drift trails.",
                "C": "Assisted motion leads, active self-propulsion is the weaker rival, and passive drift trails because the thermal evidence does too little to favor self-propulsion specifically.",
                "D": "No ordering should lead, because even repeated vessel-anchored warmth stays too equivocal to rank active, assisted, and passive motion.",
            }
            question["option_roles"] = {
                "A": "active_propulsion_overclosure",
                "B": "correct",
                "C": "assisted_motion_bias",
                "D": "underclosed_integration",
            }
            question["answer"] = "B"
        else:
            question["question"] = (
                f"{thermal_moment} over time rather than by one bright moment, "
                "which rank ordering is best supported among passive warming, residual or recent activity, and current propulsion?"
            )
            question["options"] = {
                "A": "Passive warming leads clearly, residual or recent activity is the weaker rival, and current propulsion trails.",
                "B": "Passive warming and residual or recent activity remain close enough that neither clearly outranks the other, while current propulsion trails.",
                "C": "Residual or recent activity leads, passive warming is the weaker rival, and current propulsion trails because diffuse brightness should be read operationally first.",
                "D": "No rank ordering should lead, because the evidence leaves passive warming, residual activity, and current propulsion effectively inseparable.",
            }
            question["option_roles"] = {
                "A": "passive_rank_overclosure",
                "B": "correct",
                "C": "recent_activity_overclaim",
                "D": "underclosed_integration",
            }
            question["answer"] = "B"
    elif focus == "truck_trailer_identity_persistence":
        question["question"] = (
            f"{temporal_span} as a temporal sequence rather than one ambiguous frame, "
            "which rank ordering is best supported among one degrading coupled-track account, one lead-unit-only carry-forward, "
            "and one full identity reset or split account?"
        )
        question["options"] = {
            "A": "One degrading coupled-track account leads clearly, while lead-unit-only carry-forward and full identity reset or split both fall well behind.",
            "B": "One degrading coupled-track account and one lead-unit-only carry-forward remain close enough that neither clearly outranks the other, while full identity reset or split trails.",
            "C": "One lead-unit-only carry-forward leads, because the span strips the trailing unit of too much stable evidence for a coupled account to remain ahead.",
            "D": "One full identity reset or split account leads, because the IR span fails to preserve enough ordered continuity to carry one articulated target through.",
        }
        question["option_roles"] = {
            "A": "continuity_overclosure",
            "B": "correct",
            "C": "lead_unit_overclaim",
            "D": "identity_reset_bias",
        }
        question["answer"] = "B"
    elif focus == "cross_modal_occlusion_persistence":
        question["question"] = (
            "Across the referenced IR attenuation interval, which rank ordering is best supported among one degrading continuous-track account, one weakened partial carry-forward, "
            "and one full identity reset account?"
        )
        question["options"] = {
            "A": "One degrading continuous-track account leads clearly, while weakened partial carry-forward and full identity reset both fall well behind.",
            "B": "One degrading continuous-track account and one weakened partial carry-forward remain close enough that neither clearly outranks the other, while full identity reset trails.",
            "C": "One weakened partial carry-forward leads, because the interval strips too much stable evidence for a stronger continuity account to remain in front.",
            "D": "One full identity reset account leads, because attenuation across the interval fails to preserve enough path-locked evidence to carry any target through.",
        }
        question["option_roles"] = {
            "A": "continuity_overclosure",
            "B": "correct",
            "C": "partial_track_overclaim",
            "D": "identity_reset_bias",
        }
        question["answer"] = "B"
    elif focus == "cross_modal_route_progression":
        question["question"] = (
            "When the referenced thermal trace is judged by path-consistent evolution over several frames rather than by one bright moment, "
            "which rank ordering is best supported among one same specific target, one same generic route participant, and one clutter or alternative-target account?"
        )
        question["options"] = {
            "A": "One same specific target leads clearly, one same generic route participant is the weaker rival, and clutter or an alternative target trails.",
            "B": "One same specific target and one same generic route participant remain close enough that neither clearly outranks the other, while clutter or an alternative target trails.",
            "C": "One same generic route participant leads, one same specific target is the weaker rival, and clutter or an alternative target trails because the trace carries route-locking better than exact identity.",
            "D": "Clutter or an alternative target leads, because the IR interval never becomes stable enough to keep one carried roadway participant in front.",
        }
        question["option_roles"] = {
            "A": "specific_identity_overclosure",
            "B": "correct",
            "C": "generic_chain_overclaim",
            "D": "clutter_bias",
        }
        question["answer"] = "B"
    else:
        question["question"] = (
            "When the referenced span is judged across time rather than by one salient frame, "
            "which rank ordering is best supported among a stronger carry-forward account, a weaker downgraded alternative, "
            "and a clutter, artifact, or reset account?"
        )
        question["options"] = {
            "A": "A stronger carry-forward account leads clearly, while the weaker downgraded alternative and clutter, artifact, or reset accounts both fall well behind.",
            "B": "A stronger carry-forward account and a weaker downgraded alternative remain close enough that neither clearly outranks the other, while clutter, artifact, or reset trails.",
            "C": "A weaker downgraded alternative leads, because the span carries only limited continuity pressure across time.",
            "D": "A clutter, artifact, or reset account leads, because the span never stabilizes enough to keep any carried target account in front.",
        }
        question["option_roles"] = {
            "A": "carry_forward_overclosure",
            "B": "correct",
            "C": "downgraded_continuity_overclaim",
            "D": "artifact_bias",
        }
        question["answer"] = "B"


def harden_annotation(path: Path) -> bool:
    annotation = load_json(path)
    qa = annotation["qa"]
    q2 = next(item for item in qa if item["uid"] == "2")
    q6 = next(item for item in qa if item["uid"] == "6")
    rewrite_q2(annotation, q2)
    rewrite_q6(annotation, q6)
    relabel_question(path, q2, "bulk_video_reason_q2_v1")
    relabel_question(path, q6, "bulk_video_reason_q6_v1")
    dump_json(path, annotation)
    return True


def main() -> None:
    changed = 0
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        if path.name in ALREADY_HARDENED:
            continue
        if harden_annotation(path):
            changed += 1
    print({"changed_files": changed})


if __name__ == "__main__":
    main()
