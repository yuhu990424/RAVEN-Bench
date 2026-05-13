from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import copy
import json
import re
from pathlib import Path

from tools.maintenance.rewrite_annotations_v14_deep import (
    ANNOTATIONS_DIR,
    PROTECTED,
    dump_json,
    fields,
    load_json,
    make_qa,
    normalize,
    profile_for,
    refs_by_uid,
    sent,
)


def object_label(domain: str) -> str:
    return "the object"


def cue_bank(profile: dict) -> dict[str, str]:
    domain = profile["domain"]
    target, setting, eo, ir, trap, relation, secondary = fields(profile)
    if domain == "maritime_stationary":
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "pier, berth, or terminal geometry",
            "motion": "a wake-like water disturbance",
            "thermal": "localized warm terminal or ship structure",
            "background": "fixed harbor background",
        }
    if domain.startswith("maritime"):
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "shoreline, glare, or open-water texture",
            "motion": "wake shape next to the hull",
            "thermal": "warm hull or disturbed-water contrast",
            "background": "bright water or fixed shoreline texture",
        }
    if domain == "road_stationary":
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "parking-lot lines and nearby parked vehicles",
            "motion": "a nearby moving person or vehicle",
            "thermal": "engine-area heat and cool body contrast",
            "background": "asphalt markings or curb edges",
        }
    if domain.startswith("road"):
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "lane edges, shoulders, fields, or road markings",
            "motion": "road-following motion through scale changes",
            "thermal": "vehicle heat against pavement or field background",
            "background": "roadside shadows or field texture",
        }
    if domain == "aircraft_stationary":
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "apron, stand, grass, or pavement geometry",
            "motion": "camera orbit or scale change",
            "thermal": "engine, pavement, or body thermal contrast",
            "background": "runway, taxiway, or apron markings",
        }
    if domain.startswith("aircraft"):
        return {
            "relation": relation,
            "secondary": secondary,
            "scene": "runway, taxiway, apron, or glare region",
            "motion": "taxiing or ground-motion cue",
            "thermal": "aircraft contrast against heated pavement",
            "background": "sun glare or paved-surface texture",
        }
    return {
        "relation": relation,
        "secondary": secondary,
        "scene": setting,
        "motion": "screen motion through the view",
        "thermal": "thermal contrast near the object",
        "background": "fixed background texture",
    }


def ensure_refs(annotation: dict) -> dict[str, dict[str, str]]:
    refs = refs_by_uid(annotation)
    for uid in map(str, range(1, 13)):
        refs.setdefault(uid, {"time_reference_eo": "", "time_reference_ir": ""})
    return refs


def opt(text: str) -> str:
    return normalize(text).rstrip(".")


def pair(a: str, b: str) -> str:
    return f"{opt(a)}; {opt(b)}"


def sea_plane_repair_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    eo = profile["eo"]
    ir = profile["ir"]
    trap = profile["trap"]
    return [
        make_qa(
            1,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            1,
            "L1",
            "In the early EO dock view, which object-location cue pair is visible?",
            pair("red-and-white floatplane beside a dock finger", "similar floatplanes in adjacent slips"),
            [
                ("neighbor_swap", pair("neighboring light floatplane beside the dock finger", "red-and-white aircraft farther back")),
                ("dock_swap", pair("empty dock finger in the foreground", "aircraft only as a background row")),
                ("background_swap", pair("far shoreline edge near the dock", "aircraft grouping off to one side")),
            ],
            ["spatial_reasoning", "entity_grounding"],
            "EO",
            refs["1"],
            eo,
        ),
        make_qa(
            2,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            2,
            "L2",
            "During the EO orbit, which pairing is preserved as the angle changes?",
            pair("red-and-white nose/floats", "the same dock-slip relation"),
            [
                ("neighbor_orbit", pair("neighboring light aircraft nose/floats", "the same dock-slip relation")),
                ("dock_orbit", pair("dock edge and mooring lines", "the same aircraft-like outline")),
                ("row_orbit", pair("background aircraft row", "the foreground dock-slip relation")),
            ],
            ["temporal_reasoning", "camera_motion_vs_target_motion"],
            "EO",
            refs["2"],
            eo,
        ),
        make_qa(
            3,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            3,
            "L2",
            "In IR, which local thermal relation appears at the docked aircraft row?",
            pair("one nose/cowling hotspot", "cooler neighboring aircraft bodies"),
            [
                ("neighbor_hotspot", pair("a neighboring nose/cowling hotspot", "cooler red-and-white aircraft body")),
                ("dock_hotspot", pair("a hotspot on dock hardware", "cool aircraft bodies beside it")),
                ("row_hotspot", pair("a broad warm band across several aircraft", "no isolated aircraft hotspot")),
            ],
            ["thermal_evidence_interpretation", "entity_grounding"],
            "IR",
            refs["3"],
            ir,
        ),
        make_qa(
            4,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            4,
            "L2",
            "Which nearby EO element can confuse the aircraft identity in this scene?",
            pair("adjacent docked floatplanes", "similar outlines near the selected aircraft"),
            [
                ("dock_confuser", pair("dock planks and mooring lines", "straight edges near the floats")),
                ("water_confuser", pair("water texture around the dock", "bright patches near the aircraft row")),
                ("shore_confuser", pair("far shoreline texture", "thin shapes beyond the dock")),
            ],
            ["distractor_rejection", "evidence_sufficiency"],
            "EO",
            refs["4"],
            trap,
        ),
        make_qa(
            5,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            5,
            "L2",
            "Which EO/IR association is visible across the docked seaplane sequence?",
            pair("EO: red-and-white aircraft at a dock slip", "IR: localized nose/cowling hotspot on that aircraft"),
            [
                ("neighbor_modal", pair("EO: neighboring light aircraft at an adjacent slip", "IR: localized nose/cowling hotspot on that aircraft")),
                ("dock_modal", pair("EO: dock hardware next to the aircraft row", "IR: localized hotspot on the hardware")),
                ("row_modal", pair("EO: background aircraft row", "IR: broad warm band across the row")),
            ],
            ["thermal_evidence_interpretation", "temporal_reasoning"],
            "EO+IR",
            refs["5"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            6,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            6,
            "L2",
            "Which relation continues through the camera/view change?",
            pair("red-and-white aircraft", "fixed relation to its dock finger"),
            [
                ("neighbor_relation", pair("neighboring light aircraft", "fixed relation to the same dock finger")),
                ("dock_relation", pair("dock edge", "fixed relation to an aircraft-like outline")),
                ("background_relation", pair("background aircraft row", "fixed relation to the foreground dock finger")),
            ],
            ["camera_motion_vs_target_motion", "trajectory_reasoning"],
            "EO",
            refs["6"],
            eo,
        ),
        make_qa(
            7,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            7,
            "L3",
            "Which EO object is paired with the localized IR contrast?",
            pair("red-and-white docked aircraft", "nose/cowling hotspot"),
            [
                ("neighbor_hot", pair("neighboring docked aircraft", "nose/cowling hotspot")),
                ("dock_hot", pair("dock-side hardware", "compact hotspot")),
                ("water_hot", pair("water patch beside the floats", "compact hotspot")),
            ],
            ["cross_modal_phenomenon_explanation", "evidence_sufficiency"],
            "EO+IR",
            refs["7"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            8,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            8,
            "L3",
            "Which competing cue appears near the cross-modal match?",
            pair("adjacent cold aircraft", "similar EO outline near the hotspot-bearing aircraft"),
            [
                ("dock_compete", pair("dock hardware", "straight-line EO cue near a compact hotspot")),
                ("water_compete", pair("water glare", "bright EO patch beside a compact hotspot")),
                ("shore_compete", pair("shoreline texture", "thin EO structure near a warm region")),
            ],
            ["cross_modal_phenomenon_explanation", "temporal_reasoning"],
            "EO+IR",
            refs["8"],
            trap,
        ),
        make_qa(
            9,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            1,
            "L2",
            "After the angle shift, which identity comparison remains relevant?",
            pair("red-and-white aircraft", "neighboring docked aircraft"),
            [
                ("dock_compare", pair("red-and-white aircraft", "dock hardware")),
                ("water_compare", pair("neighboring docked aircraft", "water glare")),
                ("row_compare", pair("background aircraft row", "far shoreline edge")),
            ],
            ["temporal_reasoning", "evidence_sufficiency"],
            "EO",
            refs["9"],
            eo,
        ),
        make_qa(
            10,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            2,
            "L2",
            "Which IR cue pairing appears in the selected interval?",
            pair("localized nose/cowling hotspot", "cooler aircraft beside it"),
            [
                ("neighbor_ir", pair("localized nose/cowling hotspot", "cooler red-and-white aircraft beside it")),
                ("dock_ir", pair("localized dock-edge hotspot", "cool aircraft beside it")),
                ("water_ir", pair("warm water patch near the floats", "cool aircraft row behind it")),
            ],
            ["thermal_evidence_interpretation", "spatial_reasoning"],
            "IR",
            refs["10"],
            ir,
        ),
        make_qa(
            11,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            3,
            "L3",
            "Which combined EO/IR sequence account is visible?",
            pair("EO: red-and-white aircraft keeps its dock-slip relation", "IR: compact hotspot sits on its nose/cowling"),
            [
                ("neighbor_account", pair("EO: neighboring aircraft keeps the dock-slip relation", "IR: compact hotspot sits on its nose/cowling")),
                ("dock_account", pair("EO: dock hardware keeps the foreground relation", "IR: compact hotspot sits on the dock edge")),
                ("row_account", pair("EO: background aircraft row keeps the relation", "IR: broad warm band covers the row")),
            ],
            ["cross_modal_phenomenon_explanation", "causal_reasoning"],
            "EO+IR",
            refs["11"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            12,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            4,
            "L3",
            "For the full clip, which cue chain is visible?",
            pair("red-and-white docked aircraft keeps the same slip relation", "localized nose/cowling hotspot appears on it"),
            [
                ("neighbor_verdict", pair("neighboring docked aircraft keeps the same slip relation", "localized nose/cowling hotspot appears on it")),
                ("dock_verdict", pair("dock-side hardware keeps the same foreground relation", "localized hotspot appears on it")),
                ("row_verdict", pair("background aircraft row keeps the same relation", "broad warm band appears across it")),
            ],
            ["group_verdict", "temporal_reasoning"],
            "EO+IR",
            refs["12"],
            f"EO: {eo}; IR: {ir}",
        ),
    ]


def fishing_boat_repair_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    eo = profile["eo"]
    ir = profile["ir"]
    trap = profile["trap"]
    return [
        make_qa(
            1,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            1,
            "L1",
            "In the early EO view, which local hull/water pairing is visible?",
            pair("small hull with rigging", "curved wake attached behind it"),
            [
                ("offset_wake", pair("small hull with rigging", "curved wake offset beside it")),
                ("plain_hull", pair("low plain hull", "curved wake attached behind it")),
                ("water_texture", pair("bright water texture", "rigging-like lines beside it")),
            ],
            ["spatial_reasoning", "entity_grounding"],
            "EO",
            refs["1"],
            eo,
        ),
        make_qa(
            2,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            2,
            "L2",
            "Across close and wider EO views, which cue stays tied to the vessel?",
            pair("rigging over the hull", "curved wake from the stern area"),
            [
                ("wake_detached", pair("rigging over the hull", "curved wake detached to the side")),
                ("hull_without_rigging", pair("plain hull edge", "curved wake from the stern area")),
                ("background_curve", pair("bright background curve", "hull-like patch beside it")),
            ],
            ["temporal_reasoning", "camera_motion_vs_target_motion"],
            "EO",
            refs["2"],
            eo,
        ),
        make_qa(
            3,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            3,
            "L2",
            "In IR, which thermal/wake relation appears?",
            pair("warm compact hull", "wake disturbance connected behind it"),
            [
                ("offset_ir", pair("warm compact hull", "wake disturbance offset to the side")),
                ("cool_hull", pair("cool compact hull", "warm wake disturbance behind it")),
                ("background_ir", pair("warm water streak", "hull-shaped cool patch beside it")),
            ],
            ["thermal_evidence_interpretation", "entity_grounding"],
            "IR",
            refs["3"],
            ir,
        ),
        make_qa(
            4,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            4,
            "L2",
            "Which nearby cue can be confused with the vessel path?",
            pair("curved wake segment", "near the moving hull"),
            [
                ("hull_confuser", pair("rigging over the hull", "near the wake segment")),
                ("background_confuser", pair("open-water brightness", "near the wake segment")),
                ("edge_confuser", pair("frame-edge water texture", "near the hull path")),
            ],
            ["distractor_rejection", "evidence_sufficiency"],
            "EO",
            refs["4"],
            trap,
        ),
        make_qa(
            5,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            5,
            "L2",
            "Which EO/IR cue pair is visible for the vessel track?",
            pair("EO: rigging over a small hull with curved wake", "IR: warm hull with connected wake disturbance"),
            [
                ("offset_modal", pair("EO: rigging over a small hull with offset wake", "IR: warm hull with connected wake disturbance")),
                ("plain_modal", pair("EO: plain low hull with curved wake", "IR: warm hull with connected wake disturbance")),
                ("water_modal", pair("EO: bright water curve with rigging-like lines", "IR: warm water streak beside a cool hull")),
            ],
            ["thermal_evidence_interpretation", "temporal_reasoning"],
            "EO+IR",
            refs["5"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            6,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            6,
            "L2",
            "Which pairing continues through the view change?",
            pair("rigged compact hull", "curved wake remains attached behind it"),
            [
                ("wake_side", pair("rigged compact hull", "curved wake remains off to the side")),
                ("plain_side", pair("plain compact hull", "curved wake remains attached behind it")),
                ("texture_side", pair("water-texture patch", "curved wake remains attached behind it")),
            ],
            ["camera_motion_vs_target_motion", "trajectory_reasoning"],
            "EO",
            refs["6"],
            eo,
        ),
        make_qa(
            7,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            7,
            "L3",
            "Which EO cue aligns with the IR vessel contrast?",
            pair("rigging over compact hull", "warm hull and connected wake"),
            [
                ("offset_alignment", pair("rigging over compact hull", "warm hull and offset wake")),
                ("plain_alignment", pair("plain low hull", "warm hull and connected wake")),
                ("water_alignment", pair("bright water curve", "warm streak and cool hull patch")),
            ],
            ["cross_modal_phenomenon_explanation", "evidence_sufficiency"],
            "EO+IR",
            refs["7"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            8,
            "C1",
            "consistency",
            "v15_sample_textgate_repair",
            8,
            "L3",
            "Which competing cue appears near the cross-modal vessel chain?",
            pair("curved wake segment", "close to the hull and thermal wake"),
            [
                ("rigging_compete", pair("rigging lines", "close to the hull and thermal wake")),
                ("water_compete", pair("open-water brightness", "close to the hull and thermal wake")),
                ("frame_compete", pair("frame-edge texture", "close to the hull and thermal wake")),
            ],
            ["cross_modal_phenomenon_explanation", "temporal_reasoning"],
            "EO+IR",
            refs["8"],
            trap,
        ),
        make_qa(
            9,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            1,
            "L2",
            "When the framing changes, which comparison remains necessary?",
            pair("rigged hull", "curved wake segment"),
            [
                ("plain_compare", pair("plain hull edge", "curved wake segment")),
                ("water_compare", pair("rigging-like water texture", "hull-shaped patch")),
                ("edge_compare", pair("frame-edge texture", "open-water brightness")),
            ],
            ["temporal_reasoning", "evidence_sufficiency"],
            "EO",
            refs["9"],
            eo,
        ),
        make_qa(
            10,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            2,
            "L2",
            "Which IR cue pairing appears in this interval?",
            pair("warm compact hull", "connected wake disturbance"),
            [
                ("offset_ir_pair", pair("warm compact hull", "offset wake disturbance")),
                ("cool_ir_pair", pair("cool compact hull", "warm connected wake disturbance")),
                ("background_ir_pair", pair("warm water streak", "cool hull-shaped patch")),
            ],
            ["thermal_evidence_interpretation", "spatial_reasoning"],
            "IR",
            refs["10"],
            ir,
        ),
        make_qa(
            11,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            3,
            "L3",
            "Which combined EO/IR sequence account is visible?",
            pair("EO: rigged hull remains tied to the curved wake", "IR: warm hull remains tied to the wake disturbance"),
            [
                ("offset_account", pair("EO: rigged hull remains tied to an offset wake", "IR: warm hull remains tied to the wake disturbance")),
                ("plain_account", pair("EO: plain hull remains tied to the curved wake", "IR: warm hull remains tied to the wake disturbance")),
                ("water_account", pair("EO: bright water curve remains tied to rigging-like texture", "IR: warm streak remains tied to a cool hull patch")),
            ],
            ["cross_modal_phenomenon_explanation", "causal_reasoning"],
            "EO+IR",
            refs["11"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            12,
            "H1",
            "coherence",
            "v15_sample_textgate_repair",
            4,
            "L3",
            "For the full clip, which cue chain is visible?",
            pair("rigged compact hull", "curved wake and warm-hull relation persist"),
            [
                ("offset_verdict", pair("rigged compact hull", "offset wake and warm-hull relation persist")),
                ("plain_verdict", pair("plain compact hull", "curved wake and warm-hull relation persist")),
                ("texture_verdict", pair("water-texture patch", "curved wake and warm streak persist")),
            ],
            ["group_verdict", "temporal_reasoning"],
            "EO+IR",
            refs["12"],
            f"EO: {eo}; IR: {ir}",
        ),
    ]


def _sample_ref(eo: str = "", ir: str = "") -> dict[str, str]:
    return {"time_reference_eo": eo, "time_reference_ir": ir}


def sea_plane_screen_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    eo = profile["eo"]
    ir = profile["ir"]
    trap = profile["trap"]
    return [
        make_qa(1, "C1", "consistency", "v15_screen_position_repair", 1, "L1",
                "At the start of the EO clip, which screen layout matches the red-and-white floatplane?",
                pair("tail high on the right side", "nose/front angled toward the lower-left side"),
                [("flip_horizontal", pair("tail high on the left side", "nose/front angled toward the lower-right side")),
                 ("flip_vertical", pair("tail low on the right side", "nose/front angled toward the upper-left side")),
                 ("neighbor_layout", pair("white neighbor low on the left side", "red aircraft only in the upper background"))],
                ["spatial_reasoning", "entity_grounding"], "EO", _sample_ref("00:00-00:04", ""), eo),
        make_qa(2, "C1", "consistency", "v15_screen_position_repair", 2, "L2",
                "Across the high EO orbit, which relative position stays associated with the red aircraft?",
                pair("red aircraft at the lower end of the docked-aircraft cluster", "paler aircraft above or beside it"),
                [("row_top", pair("red aircraft at the upper end of the docked-aircraft cluster", "paler aircraft below it")),
                 ("row_middle", pair("red aircraft in the middle of the pale-aircraft row", "no lower-end separation")),
                 ("shore_swap", pair("red aircraft near the shoreline road", "paler aircraft at the dock end"))],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", _sample_ref("00:08-00:24", ""), eo),
        make_qa(3, "C1", "consistency", "v15_screen_position_repair", 3, "L2",
                "In the matching IR orbit, which local aircraft-row relation is visible?",
                pair("compact warm return near the lower aircraft position", "cooler aircraft shapes above it"),
                [("upper_warm", pair("compact warm return near the upper aircraft position", "cooler aircraft shapes below it")),
                 ("dock_warm", pair("compact warm return on the dock finger", "cool aircraft shapes on both sides")),
                 ("row_warm", pair("continuous warm strip across the aircraft row", "no compact lower-position return"))],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", _sample_ref("", "00:24-00:40"), ir),
        make_qa(4, "C1", "consistency", "v15_screen_position_repair", 4, "L2",
                "Which nearby EO element is the closest same-scene aircraft distractor?",
                pair("pale floatplane beside the red aircraft", "same dock row"),
                [("dock_distractor", pair("straight dock finger beside the red aircraft", "same dock row")),
                 ("water_distractor", pair("sparkling water beside the red aircraft", "same dock row")),
                 ("shore_distractor", pair("curved shoreline road beside the aircraft row", "same dock row"))],
                ["distractor_rejection", "evidence_sufficiency"], "EO", _sample_ref("00:24-00:40", ""), trap),
        make_qa(5, "C1", "consistency", "v15_screen_position_repair", 5, "L2",
                "Which EO/IR position match is visible around the docked aircraft?",
                pair("EO: red aircraft at the lower/right side of the cluster", "IR: compact warm return at the matching lower/right position"),
                [("upper_match", pair("EO: red aircraft at the upper/left side of the cluster", "IR: compact warm return at the matching upper/left position")),
                 ("neighbor_match", pair("EO: pale neighbor at the lower/right side of the cluster", "IR: compact warm return at the matching lower/right position")),
                 ("dock_match", pair("EO: dock finger at the lower/right side of the cluster", "IR: compact warm return at the matching lower/right position"))],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", _sample_ref("00:32-00:40", "00:32-00:40"), f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "v15_screen_position_repair", 6, "L2",
                "In the close EO pass, which orientation remains visible?",
                pair("red H-marked tail on the screen-right side", "nose/cowling toward the screen-left side"),
                [("left_tail", pair("red H-marked tail on the screen-left side", "nose/cowling toward the screen-right side")),
                 ("top_tail", pair("red H-marked tail near the screen top", "nose/cowling toward the screen bottom")),
                 ("bottom_tail", pair("red H-marked tail near the screen bottom", "nose/cowling toward the screen top"))],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", _sample_ref("01:13-01:29", ""), eo),
        make_qa(7, "C1", "consistency", "v15_screen_position_repair", 7, "L3",
                "Which close-pass EO/IR orientation match is visible?",
                pair("EO: tail to the right and nose to the left", "IR: warm front area on the left-side end"),
                [("reverse_front", pair("EO: tail to the left and nose to the right", "IR: warm front area on the right-side end")),
                 ("top_front", pair("EO: tail at the bottom and nose at the top", "IR: warm front area on the top end")),
                 ("neighbor_front", pair("EO: pale neighbor to the left and red aircraft to the right", "IR: warm front area on the neighbor"))],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "v15_screen_position_repair", 8, "L3",
                "Which competing cue remains near the close-pass match?",
                pair("pale neighboring aircraft at the left side of the red aircraft", "same dock line"),
                [("right_neighbor", pair("pale neighboring aircraft at the right side of the red aircraft", "same dock line")),
                 ("upper_neighbor", pair("pale neighboring aircraft above the red aircraft", "same dock line")),
                 ("lower_neighbor", pair("pale neighboring aircraft below the red aircraft", "same dock line"))],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), trap),
        make_qa(9, "H1", "coherence", "v15_screen_position_repair", 1, "L2",
                "After the view tightens, which EO comparison keeps the aircraft identity grounded?",
                pair("screen-right red tail", "screen-left nose/front"),
                [("reverse_compare", pair("screen-left red tail", "screen-right nose/front")),
                 ("vertical_compare", pair("screen-top red tail", "screen-bottom nose/front")),
                 ("dock_compare", pair("screen-right dock edge", "screen-left aircraft row"))],
                ["temporal_reasoning", "evidence_sufficiency"], "EO", _sample_ref("01:13-01:29", ""), eo),
        make_qa(10, "H1", "coherence", "v15_screen_position_repair", 2, "L2",
                "Which IR screen-side relation appears in the close interval?",
                pair("warmer front area toward the left end", "tail-side structure cooler toward the right end"),
                [("right_front", pair("warmer front area toward the right end", "tail-side structure cooler toward the left end")),
                 ("top_front", pair("warmer front area toward the top end", "tail-side structure cooler toward the bottom end")),
                 ("dock_front", pair("warmer front area on the dock edge", "aircraft body cooler above it"))],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", _sample_ref("", "01:13-01:29"), ir),
        make_qa(11, "H1", "coherence", "v15_screen_position_repair", 3, "L3",
                "Which combined sequence account follows the same screen orientation?",
                pair("EO: red tail stays on the right in the close pass", "IR: front warmth is on the left-side end"),
                [("reverse_account", pair("EO: red tail stays on the left in the close pass", "IR: front warmth is on the right-side end")),
                 ("vertical_account", pair("EO: red tail stays at the bottom in the close pass", "IR: front warmth is on the top end")),
                 ("neighbor_account", pair("EO: pale neighbor takes the right-side tail position", "IR: front warmth is on that neighbor"))],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "v15_screen_position_repair", 4, "L3",
                "For the full clip, which docked-aircraft cue chain is visible?",
                pair("wide orbit: red aircraft at the lower dock-row end", "close pass: tail right and nose/front left"),
                [("upper_verdict", pair("wide orbit: red aircraft at the upper dock-row end", "close pass: tail left and nose/front right")),
                 ("middle_verdict", pair("wide orbit: red aircraft in the middle of the row", "close pass: tail bottom and nose/front top")),
                 ("neighbor_verdict", pair("wide orbit: pale neighbor at the lower dock-row end", "close pass: red aircraft only in background"))],
                ["group_verdict", "temporal_reasoning"], "EO+IR", _sample_ref("00:08-01:29", "00:08-01:29"), f"EO: {eo}; IR: {ir}"),
    ]


def fishing_boat_screen_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    eo = profile["eo"]
    ir = profile["ir"]
    trap = profile["trap"]
    return [
        make_qa(1, "C1", "consistency", "v15_screen_position_repair", 1, "L1",
                "Around 00:10 EO, which screen-side hull/wake relation is visible?",
                pair("hull on the left side of the wake", "white wake extending to the right"),
                [("left_wake", pair("hull on the right side of the wake", "white wake extending to the left")),
                 ("upper_wake", pair("hull below the wake", "white wake extending upward")),
                 ("lower_wake", pair("hull above the wake", "white wake extending downward"))],
                ["spatial_reasoning", "entity_grounding"], "EO", _sample_ref("00:08-00:14", ""), eo),
        make_qa(2, "C1", "consistency", "v15_screen_position_repair", 2, "L2",
                "Around 00:30 EO, which motion layout is visible?",
                pair("boat at the left/front of the bright trail", "wake trailing toward screen right"),
                [("right_front", pair("boat at the right/front of the bright trail", "wake trailing toward screen left")),
                 ("top_front", pair("boat at the top/front of the bright trail", "wake trailing toward screen bottom")),
                 ("bottom_front", pair("boat at the bottom/front of the bright trail", "wake trailing toward screen top"))],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", _sample_ref("00:28-00:33", ""), eo),
        make_qa(3, "C1", "consistency", "v15_screen_position_repair", 3, "L2",
                "Around 00:30 IR, which hull/wake thermal layout is visible?",
                pair("bright hull near the right side", "broader wake texture spreading leftward"),
                [("left_hull", pair("bright hull near the left side", "broader wake texture spreading rightward")),
                 ("top_hull", pair("bright hull near the top side", "broader wake texture spreading downward")),
                 ("bottom_hull", pair("bright hull near the bottom side", "broader wake texture spreading upward"))],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", _sample_ref("", "00:28-00:33"), ir),
        make_qa(4, "C1", "consistency", "v15_screen_position_repair", 4, "L2",
                "Around 00:50 EO, which V-wake layout creates the nearby path cue?",
                pair("boat at the lower-left end", "wake arms opening toward the upper-right side"),
                [("lower_right", pair("boat at the lower-right end", "wake arms opening toward the upper-left side")),
                 ("upper_left", pair("boat at the upper-left end", "wake arms opening toward the lower-right side")),
                 ("upper_right", pair("boat at the upper-right end", "wake arms opening toward the lower-left side"))],
                ["distractor_rejection", "evidence_sufficiency"], "EO", _sample_ref("00:48-00:53", ""), trap),
        make_qa(5, "C1", "consistency", "v15_screen_position_repair", 5, "L2",
                "Around 00:50, which EO/IR wake-position match is visible?",
                pair("EO: hull at lower-left with wake opening upper-right", "IR: bright hull at lower-left with textured wake upper-right"),
                [("eo_ir_lr", pair("EO: hull at lower-right with wake opening upper-left", "IR: bright hull at lower-right with textured wake upper-left")),
                 ("eo_ir_ul", pair("EO: hull at upper-left with wake opening lower-right", "IR: bright hull at upper-left with textured wake lower-right")),
                 ("eo_ir_ur", pair("EO: hull at upper-right with wake opening lower-left", "IR: bright hull at upper-right with textured wake lower-left"))],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", _sample_ref("00:48-00:53", "00:48-00:53"), f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "v15_screen_position_repair", 6, "L2",
                "Around 01:00 EO, which framing relation appears?",
                pair("boat near the lower center", "two wake bands extending upward behind it"),
                [("upper_center", pair("boat near the upper center", "two wake bands extending downward behind it")),
                 ("left_center", pair("boat near the left center", "two wake bands extending rightward behind it")),
                 ("right_center", pair("boat near the right center", "two wake bands extending leftward behind it"))],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", _sample_ref("00:58-01:03", ""), eo),
        make_qa(7, "C1", "consistency", "v15_screen_position_repair", 7, "L3",
                "Around 01:00, which EO/IR direction match is visible?",
                pair("EO: boat lower in frame with wake above it", "IR: bright hull lower in frame with wake texture above it"),
                [("upper_match", pair("EO: boat upper in frame with wake below it", "IR: bright hull upper in frame with wake texture below it")),
                 ("left_match", pair("EO: boat left in frame with wake to the right", "IR: bright hull left in frame with wake texture to the right")),
                 ("right_match", pair("EO: boat right in frame with wake to the left", "IR: bright hull right in frame with wake texture to the left"))],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", _sample_ref("00:58-01:03", "00:58-01:03"), f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "v15_screen_position_repair", 8, "L3",
                "Around 01:30 EO, which close-pass screen orientation is visible?",
                pair("bow/front points to the right", "white wake exits to the left"),
                [("left_bow", pair("bow/front points to the left", "white wake exits to the right")),
                 ("up_bow", pair("bow/front points upward", "white wake exits downward")),
                 ("down_bow", pair("bow/front points downward", "white wake exits upward"))],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", _sample_ref("01:28-01:33", "01:28-01:33"), trap),
        make_qa(9, "H1", "coherence", "v15_screen_position_repair", 1, "L2",
                "Around 01:30 IR, which side-on thermal relation is visible?",
                pair("bright hull profile toward the right", "darker wake trail toward the left"),
                [("left_profile", pair("bright hull profile toward the left", "darker wake trail toward the right")),
                 ("top_profile", pair("bright hull profile toward the top", "darker wake trail toward the bottom")),
                 ("bottom_profile", pair("bright hull profile toward the bottom", "darker wake trail toward the top"))],
                ["temporal_reasoning", "evidence_sufficiency"], "IR", _sample_ref("", "01:28-01:33"), ir),
        make_qa(10, "H1", "coherence", "v15_screen_position_repair", 2, "L2",
                "Around 01:40 EO, which direction relation appears?",
                pair("boat heading toward the upper-right", "wake trailing lower-left"),
                [("upper_left", pair("boat heading toward the upper-left", "wake trailing lower-right")),
                 ("lower_right", pair("boat heading toward the lower-right", "wake trailing upper-left")),
                 ("lower_left", pair("boat heading toward the lower-left", "wake trailing upper-right"))],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "EO", _sample_ref("01:38-01:43", ""), eo),
        make_qa(11, "H1", "coherence", "v15_screen_position_repair", 3, "L3",
                "Around 01:40, which combined EO/IR direction account is visible?",
                pair("EO: boat ahead toward upper-right", "IR: bright hull ahead of darker strands to lower-left"),
                [("ul_account", pair("EO: boat ahead toward upper-left", "IR: bright hull ahead of darker strands to lower-right")),
                 ("lr_account", pair("EO: boat ahead toward lower-right", "IR: bright hull ahead of darker strands to upper-left")),
                 ("ll_account", pair("EO: boat ahead toward lower-left", "IR: bright hull ahead of darker strands to upper-right"))],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", _sample_ref("01:38-01:43", "01:38-01:43"), f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "v15_screen_position_repair", 4, "L3",
                "At the end of the clip, which full-frame cue chain is visible?",
                pair("boat near the upper center", "wake column extending downward through the frame"),
                [("bottom_end", pair("boat near the lower center", "wake column extending upward through the frame")),
                 ("left_end", pair("boat near the left center", "wake column extending rightward through the frame")),
                 ("right_end", pair("boat near the right center", "wake column extending leftward through the frame"))],
                ["group_verdict", "temporal_reasoning"], "EO+IR", _sample_ref("01:48-01:51", "01:48-01:51"), f"EO: {eo}; IR: {ir}"),
    ]


def sea_plane_screen_questions_v2(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    eo = profile["eo"]
    ir = profile["ir"]
    trap = profile["trap"]
    return [
        make_qa(1, "C1", "consistency", "v15_screen_position_repair_v2", 1, "L1",
                "At the first close EO view, which nearby layout is visible around the red-and-white aircraft?",
                pair("partial pale floatplane at the lower-left edge", "dock structure along the right/lower side"),
                [("upper_left_neighbor", pair("partial pale floatplane at the upper-left edge", "dock structure along the right/lower side")),
                 ("lower_right_neighbor", pair("partial pale floatplane at the lower-right edge", "dock structure along the left/lower side")),
                 ("upper_right_neighbor", pair("partial pale floatplane at the upper-right edge", "dock structure along the left/upper side"))],
                ["spatial_reasoning", "entity_grounding"], "EO", _sample_ref("00:00-00:04", ""), eo),
        make_qa(2, "C1", "consistency", "v15_screen_position_repair_v2", 2, "L2",
                "In the high EO orbit, which dock-row placement is visible for the red aircraft?",
                pair("red aircraft at the lower end of the row", "pale aircraft clustered above it"),
                [("upper_row", pair("red aircraft at the upper end of the row", "pale aircraft clustered below it")),
                 ("left_row", pair("red aircraft at the left end of the row", "pale aircraft clustered to its right")),
                 ("right_row", pair("red aircraft at the right end of the row", "pale aircraft clustered to its left"))],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", _sample_ref("00:08-00:24", ""), eo),
        make_qa(3, "C1", "consistency", "v15_screen_position_repair_v2", 3, "L2",
                "In the matching IR wide view, where is the compact aircraft return relative to the row?",
                pair("lower end of the aircraft row", "cooler row shapes above it"),
                [("upper_ir_row", pair("upper end of the aircraft row", "cooler row shapes below it")),
                 ("left_ir_row", pair("left end of the aircraft row", "cooler row shapes to its right")),
                 ("right_ir_row", pair("right end of the aircraft row", "cooler row shapes to its left"))],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", _sample_ref("", "00:24-00:40"), ir),
        make_qa(4, "C1", "consistency", "v15_screen_position_repair_v2", 4, "L2",
                "Around 00:32 EO, which dock/water relation sits around the red aircraft?",
                pair("dock walkway above the red aircraft", "dark water below it"),
                [("dock_below", pair("dock walkway below the red aircraft", "dark water above it")),
                 ("dock_left", pair("dock walkway left of the red aircraft", "dark water right of it")),
                 ("dock_right", pair("dock walkway right of the red aircraft", "dark water left of it"))],
                ["distractor_rejection", "evidence_sufficiency"], "EO", _sample_ref("00:30-00:35", ""), trap),
        make_qa(5, "C1", "consistency", "v15_screen_position_repair_v2", 5, "L2",
                "Around 00:40, which EO/IR dock-side match is visible?",
                pair("EO: dock/walkway above-left of the red aircraft", "IR: dock/walkway return above-left of the aircraft return"),
                [("above_right_match", pair("EO: dock/walkway above-right of the red aircraft", "IR: dock/walkway return above-right of the aircraft return")),
                 ("below_left_match", pair("EO: dock/walkway below-left of the red aircraft", "IR: dock/walkway return below-left of the aircraft return")),
                 ("below_right_match", pair("EO: dock/walkway below-right of the red aircraft", "IR: dock/walkway return below-right of the aircraft return"))],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", _sample_ref("00:38-00:42", "00:38-00:42"), f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "v15_screen_position_repair_v2", 6, "L2",
                "In the late close EO pass, which background placement stays with the aircraft?",
                pair("dock edge below the aircraft", "open water above it"),
                [("dock_above", pair("dock edge above the aircraft", "open water below it")),
                 ("dock_left", pair("dock edge left of the aircraft", "open water right of it")),
                 ("dock_right", pair("dock edge right of the aircraft", "open water left of it"))],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", _sample_ref("01:13-01:29", ""), eo),
        make_qa(7, "C1", "consistency", "v15_screen_position_repair_v2", 7, "L3",
                "In the late close EO/IR interval, which background-side match is visible?",
                pair("EO: dock below and water above", "IR: bright dock band below and darker water texture above"),
                [("reverse_vertical", pair("EO: dock above and water below", "IR: bright dock band above and darker water texture below")),
                 ("left_right", pair("EO: dock left and water right", "IR: bright dock band left and darker water texture right")),
                 ("right_left", pair("EO: dock right and water left", "IR: bright dock band right and darker water texture left"))],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "v15_screen_position_repair_v2", 8, "L3",
                "Which competing cue appears near the cross-modal match?",
                pair("adjacent cold aircraft", "similar EO outline near the hotspot-bearing aircraft"),
                [("dock_compete", pair("dock hardware", "straight-line EO cue near a compact hotspot")),
                 ("water_compete", pair("water glare", "bright EO patch beside a compact hotspot")),
                 ("shore_compete", pair("shoreline texture", "thin EO structure near a warm region"))],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), trap),
        make_qa(9, "H1", "coherence", "v15_screen_position_repair_v2", 1, "L2",
                "After the view tightens, which EO background comparison is visible?",
                pair("aircraft above the dock-edge line", "water beyond the upper side of the aircraft"),
                [("below_dockline", pair("aircraft below the dock-edge line", "water beyond the lower side of the aircraft")),
                 ("left_dockline", pair("aircraft left of the dock-edge line", "water beyond the left side of the aircraft")),
                 ("right_dockline", pair("aircraft right of the dock-edge line", "water beyond the right side of the aircraft"))],
                ["temporal_reasoning", "evidence_sufficiency"], "EO", _sample_ref("01:13-01:29", ""), eo),
        make_qa(10, "H1", "coherence", "v15_screen_position_repair_v2", 2, "L2",
                "Which IR background relation appears in the same close interval?",
                pair("aircraft return just above the bright dock band", "water texture beyond the upper side"),
                [("below_ir_band", pair("aircraft return just below the bright dock band", "water texture beyond the lower side")),
                 ("left_ir_band", pair("aircraft return just left of the bright dock band", "water texture beyond the left side")),
                 ("right_ir_band", pair("aircraft return just right of the bright dock band", "water texture beyond the right side"))],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", _sample_ref("", "01:13-01:29"), ir),
        make_qa(11, "H1", "coherence", "v15_screen_position_repair_v2", 3, "L3",
                "Which combined EO/IR background sequence stays aligned?",
                pair("EO: dock below the aircraft and water above", "IR: bright dock band below the return and water texture above"),
                [("vertical_swap", pair("EO: dock above the aircraft and water below", "IR: bright dock band above the return and water texture below")),
                 ("horizontal_left", pair("EO: dock left of the aircraft and water right", "IR: bright dock band left of the return and water texture right")),
                 ("horizontal_right", pair("EO: dock right of the aircraft and water left", "IR: bright dock band right of the return and water texture left"))],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", _sample_ref("01:13-01:29", "01:13-01:29"), f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "v15_screen_position_repair_v2", 4, "L3",
                "For the full clip, which two-stage position chain is visible?",
                pair("wide view: red aircraft at lower row end", "close view: aircraft above dock line with water beyond upper side"),
                [("upper_to_below", pair("wide view: red aircraft at upper row end", "close view: aircraft below dock line with water beyond lower side")),
                 ("left_to_left", pair("wide view: red aircraft at left row end", "close view: aircraft left of dock line with water beyond left side")),
                 ("right_to_right", pair("wide view: red aircraft at right row end", "close view: aircraft right of dock line with water beyond right side"))],
                ["group_verdict", "temporal_reasoning"], "EO+IR", _sample_ref("00:08-01:29", "00:08-01:29"), f"EO: {eo}; IR: {ir}"),
    ]


def sample_repair_questions(path: Path, annotation: dict, refs: dict[str, dict[str, str]], profile: dict) -> list[dict] | None:
    sample_id = str(annotation.get("sample_id") or path.stem)
    if sample_id == "Sea.Plane":
        return sea_plane_screen_questions_v2(profile, refs)
    if sample_id == "Fishing.Boat":
        return fishing_boat_screen_questions(profile, refs)
    return None


def build_balanced_questions(path: Path, annotation: dict) -> list[dict]:
    refs = ensure_refs(annotation)
    profile = profile_for(path, annotation)
    repaired = sample_repair_questions(path, annotation, refs, profile)
    if repaired is not None:
        return repaired
    target, setting, eo, ir, trap, relation, secondary = fields(profile)
    domain = profile["domain"]
    obj = object_label(domain)
    cues = cue_bank(profile)
    stationary = domain.endswith("stationary") or domain in {"road_stationary", "maritime_stationary", "aircraft_stationary"}

    state_word = "fixed-state" if stationary else "motion"
    eo_relation = cues["relation"]
    sec = cues["secondary"]
    scene = cues["scene"]
    motion = cues["motion"]
    thermal = cues["thermal"]
    background = cues["background"]

    return [
        make_qa(
            1,
            "C1",
            "consistency",
            "v15_cue_chain",
            1,
            "L1",
            f"In the early EO segment, which cue pairing is visible for {obj}?",
            pair(f"{obj} with {eo_relation}", f"{sec} nearby"),
            [
                ("relation_swap", pair(f"{obj} with {sec}", f"{eo_relation} nearby")),
                ("background_pair", pair(f"{obj} with {background}", f"{motion} nearby")),
                ("scene_pair", pair(f"{obj} with {scene}", f"{background} nearby")),
            ],
            ["spatial_reasoning", "entity_grounding"],
            "EO",
            refs["1"],
            eo,
        ),
        make_qa(
            2,
            "C1",
            "consistency",
            "v15_cue_chain",
            2,
            "L2",
            f"Across the EO scale or angle change, which {state_word} cue stays paired with {obj}?",
            pair(eo_relation, "view scale or angle changes"),
            [
                ("secondary_takeover", pair(sec, "view scale or angle changes")),
                ("background_takeover", pair(background, "view scale or angle changes")),
                ("scene_reset", pair(scene, "view scale or angle changes")),
            ],
            ["temporal_reasoning", "camera_motion_vs_target_motion"],
            "EO",
            refs["2"],
            eo,
        ),
        make_qa(
            3,
            "C1",
            "consistency",
            "v15_cue_chain",
            3,
            "L2",
            f"In IR, which contrast pairing is visible near {obj}?",
            pair(thermal, eo_relation),
            [
                ("secondary_heat", pair(thermal, sec)),
                ("background_heat", pair(thermal, background)),
                ("scene_heat", pair(thermal, scene)),
            ],
            ["thermal_evidence_interpretation", "entity_grounding"],
            "IR",
            refs["3"],
            ir,
        ),
        make_qa(
            4,
            "C1",
            "consistency",
            "v15_cue_chain",
            4,
            "L2",
            "Which nearby cue creates a same-scene confusion?",
            pair(sec, f"near {obj}"),
            [
                ("relation_confuser", pair(eo_relation, f"near {obj}")),
                ("scene_confuser", pair(scene, f"near {obj}")),
                ("background_confuser", pair(background, f"near {obj}")),
            ],
            ["distractor_rejection", "evidence_sufficiency"],
            "EO",
            refs["4"],
            trap,
        ),
        make_qa(
            5,
            "C1",
            "consistency",
            "v15_cue_chain",
            5,
            "L2",
            "Which EO/IR cue pair is visible?",
            pair(f"EO: {eo_relation}", f"IR: {thermal}"),
            [
                ("modal_swap", pair(f"EO: {sec}", f"IR: {thermal}")),
                ("ir_background", pair(f"EO: {eo_relation}", f"IR: {background}")),
                ("scene_pair", pair(f"EO: {scene}", f"IR: {thermal}")),
            ],
            ["thermal_evidence_interpretation", "temporal_reasoning"],
            "EO+IR",
            refs["5"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            6,
            "C1",
            "consistency",
            "v15_cue_chain",
            6,
            "L2",
            "Which cue pairing continues through camera/view change?",
            pair(f"{obj}: {eo_relation}", "screen scale or angle changes"),
            [
                ("secondary_motion", pair(f"{obj}: {sec}", "screen scale or angle changes")),
                ("background_motion", pair(f"{obj}: {background}", "screen scale or angle changes")),
                ("scene_motion", pair(f"{obj}: {scene}", "screen scale or angle changes")),
            ],
            ["camera_motion_vs_target_motion", "trajectory_reasoning"],
            "EO",
            refs["6"],
            eo,
        ),
        make_qa(
            7,
            "C1",
            "consistency",
            "v15_cue_chain",
            7,
            "L3",
            "Which EO cue is aligned with the IR contrast cue?",
            pair(f"EO cue: {eo_relation}", f"IR cue: {thermal}"),
            [
                ("secondary_pair", pair(f"EO cue: {sec}", f"IR cue: {thermal}")),
                ("background_pair", pair(f"EO cue: {background}", f"IR cue: {thermal}")),
                ("scene_pair", pair(f"EO cue: {scene}", f"IR cue: {thermal}")),
            ],
            ["cross_modal_phenomenon_explanation", "evidence_sufficiency"],
            "EO+IR",
            refs["7"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            8,
            "C1",
            "consistency",
            "v15_cue_chain",
            8,
            "L3",
            "Which competing cue appears in the cross-modal chain?",
            pair(f"EO/visible cue: {sec}", f"IR/contrast context: {thermal}"),
            [
                ("target_chain", pair(f"EO/visible cue: {eo_relation}", f"IR/contrast context: {thermal}")),
                ("background_chain", pair(f"EO/visible cue: {background}", f"IR/contrast context: {thermal}")),
                ("scene_chain", pair(f"EO/visible cue: {scene}", f"IR/contrast context: {thermal}")),
            ],
            ["cross_modal_phenomenon_explanation", "temporal_reasoning"],
            "EO+IR",
            refs["8"],
            trap,
        ),
        make_qa(
            9,
            "H1",
            "coherence",
            "v15_temporal_cue_chain",
            1,
            "L2",
            "When the view changes, which cue is checked against the nearby alternative?",
            pair(eo_relation, sec),
            [
                ("secondary_first", pair(sec, eo_relation)),
                ("background_first", pair(background, eo_relation)),
                ("scene_first", pair(scene, sec)),
            ],
            ["temporal_reasoning", "evidence_sufficiency"],
            "EO",
            refs["9"],
            eo,
        ),
        make_qa(
            10,
            "H1",
            "coherence",
            "v15_temporal_cue_chain",
            2,
            "L2",
            "Which IR cue pairing appears in this interval?",
            pair(thermal, eo_relation),
            [
                ("secondary_thermal", pair(thermal, sec)),
                ("background_thermal", pair(thermal, background)),
                ("scene_thermal", pair(thermal, scene)),
            ],
            ["thermal_evidence_interpretation", "spatial_reasoning"],
            "IR",
            refs["10"],
            ir,
        ),
        make_qa(
            11,
            "H1",
            "coherence",
            "v15_temporal_cue_chain",
            3,
            "L3",
            "Which combined EO/IR sequence account is visible?",
            pair(f"EO: {eo_relation}", f"IR: {thermal}"),
            [
                ("secondary_modal", pair(f"EO: {sec}", f"IR: {thermal}")),
                ("background_modal", pair(f"EO: {background}", f"IR: {thermal}")),
                ("scene_modal", pair(f"EO: {scene}", f"IR: {thermal}")),
            ],
            ["cross_modal_phenomenon_explanation", "causal_reasoning"],
            "EO+IR",
            refs["11"],
            f"EO: {eo}; IR: {ir}",
        ),
        make_qa(
            12,
            "H1",
            "coherence",
            "v15_temporal_cue_chain",
            4,
            "L3",
            "For the full clip, which cue chain is visible?",
            pair(f"{obj}: {eo_relation}", f"nearby cue: {sec}"),
            [
                ("secondary_verdict", pair(f"{obj}: {sec}", f"nearby cue: {eo_relation}")),
                ("background_verdict", pair(f"{obj}: {background}", f"nearby cue: {sec}")),
                ("scene_verdict", pair(f"{obj}: {scene}", f"nearby cue: {eo_relation}")),
            ],
            ["group_verdict", "temporal_reasoning"],
            "EO+IR",
            refs["12"],
            trap,
        ),
    ]


def update_notes(annotation: dict) -> None:
    notes = copy.deepcopy(annotation.get("notes", {}))
    notes["design_revision_note"] = (
        "Video-reasoning v15 text-gate repair r2: options are neutral cue-pair "
        "choices with same-scene distractors, reducing answerability from wording alone."
    )
    notes["annotation_visibility_warning"] = (
        "Do not export main_event, event_description, answer, option_roles, evidence_note, "
        "rationales, or time_reference metadata to model prompts."
    )
    notes["acceptance_gates"] = [
        "Run prepare-dataset schema validation.",
        "Run request key-path leakage audit after export.",
        "Run Qwen text-only/no-video baseline before video inference.",
        "Reject samples or templates whose text-only accuracy suggests wording shortcuts.",
        "Keep evidence_note fields internal; they are for annotation audit only.",
    ]
    notes["group_scoring_recommendation"] = {
        "C1": "Use squared-ratio consistency scoring for the eight-step parallel cue chain.",
        "H1": "Prefer prefix scoring because the four questions form a cue-chain verdict.",
    }
    annotation["notes"] = notes


def validate_text(annotation: dict, path: Path) -> None:
    banned_patterns = [
        r"\bshortcut\b",
        r"\bbenchmark\b",
        r"\bmost likely\b",
        r"\bbest\b",
        r"\bsafer\b",
        r"\bshould not\b",
        r"\brather than\b",
    ]
    for question in annotation["qa"]:
        texts = [question["question"], *question["options"].values()]
        blob = "\n".join(texts).lower()
        for pattern in banned_patterns:
            if re.search(pattern, blob):
                raise ValueError(f"{path.name} uid={question['uid']} contains shortcut-prone wording: {pattern}")


def main() -> None:
    changed = []
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        if path.name in PROTECTED:
            continue
        annotation = load_json(path)
        annotation["qa"] = build_balanced_questions(path, annotation)
        update_notes(annotation)
        validate_text(annotation, path)
        dump_json(path, annotation)
        changed.append(path.name)
    print(json.dumps({"status": "ok", "changed_count": len(changed), "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
