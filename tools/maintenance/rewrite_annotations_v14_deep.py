from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = ROOT_DIR
ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
PROTECTED = {
    "5th.Wheel.json",
    "Airplane.json",
    "Air_Canada_Airliner.json",
}
LABELS = ["A", "B", "C", "D"]


PROFILE_OVERRIDES = {
    "Automobile.json": {
        "domain": "road",
        "target": "the early dark car",
        "setting": "rural two-lane roads with fields, trees, and houses",
        "eo": "the dark car follows the road before a later white van appears in a similar rural-road setting",
        "ir": "IR helps locate the early road vehicle, while the later white van should remain a separate event",
        "trap": "merging the early dark car and the later white van into one identity chain",
        "relation": "road-following continuity through scale changes",
        "secondary": "the later white van",
    },
    "Cargo.Ship_Horizon.json": {
        "domain": "maritime",
        "target": "the large container ship",
        "setting": "open water with low-angle sunlight and horizon-level views",
        "eo": "the hull keeps forward motion with a wake through distant and closer views",
        "ir": "localized hot exhaust and a warmer propulsion-side structure stay tied to the same vessel",
        "trap": "treating sun glint or the distant horizon view as a separate target",
        "relation": "hull, wake, and exhaust alignment",
        "secondary": "bright glint patches on the water",
    },
    "Cargo.Truck.2.json": {
        "domain": "road",
        "target": "the yellow or orange tractor with a long white box trailer",
        "setting": "a highway corridor seen from close oblique and wide overhead views",
        "eo": "the tractor-trailer pairing persists despite edge occlusion and strong scale changes",
        "ir": "the warmer tractor and cooler trailer remain linked during long-range tracking",
        "trap": "dropping the trailer or switching to a same-lane vehicle during scale changes",
        "relation": "tractor-to-trailer coupling",
        "secondary": "nearby highway traffic",
    },
    "Cargo.Truck.json": {
        "domain": "road",
        "target": "the white articulated semi-truck with a long box trailer",
        "setting": "a divided highway with surrounding traffic and large viewpoint changes",
        "eo": "the white tractor-trailer remains a coupled road object as the camera widens",
        "ir": "thermal contrast separates the moving tractor-trailer from road surface and traffic",
        "trap": "mistaking camera geometry or adjacent traffic for a target change",
        "relation": "box-trailer continuity across viewpoints",
        "secondary": "adjacent highway traffic",
    },
    "Container.Ship.json": {
        "domain": "maritime_stationary",
        "target": "the docked container ship",
        "setting": "a port terminal with gantry cranes, container stacks, and later terminal-like IR footage",
        "eo": "the container ship remains berth-bound next to cranes without a main wake",
        "ir": "terminal heat and crane structure appear around the berth, then a later IR segment shifts scene",
        "trap": "merging the later ferry- or terminal-like IR segment with the earlier docked ship event",
        "relation": "fixed berth relation to cranes",
        "secondary": "the later IR scene segment",
    },
    "Covered.Boat.json": {
        "domain": "maritime",
        "target": "the small passenger boat with a bright canopy",
        "setting": "open water viewed from an airborne camera",
        "eo": "the canopy and passengers under it are visible through the open sides",
        "ir": "a rear engine hotspot, cool canopy, and warmer interior passengers separate different parts of the boat",
        "trap": "reading the bright canopy as the heat source or ignoring the passengers under it",
        "relation": "canopy, passengers, and stern engine layout",
        "secondary": "the canopy brightness",
    },
    "Docked Ferry.2.json": {
        "domain": "maritime_stationary",
        "target": "the docked BC Ferries vessel",
        "setting": "a multi-slip ferry terminal near forested islands",
        "eo": "the ferry stays inside the docking structure",
        "ir": "a hot exhaust plume and stern-side water disturbance indicate active power while docked",
        "trap": "turning thermal activity at the stern into an underway departure",
        "relation": "its terminal slip",
        "secondary": "the terminal slip geometry",
    },
    "Docked_Ferry.1.json": {
        "domain": "maritime_stationary",
        "target": "the large ferry moored at the terminal",
        "setting": "a ferry terminal with piers, slips, shoreline structures, and small harbor craft",
        "eo": "the ferry remains aligned with the terminal while camera scale and angle change",
        "ir": "warm ship and terminal structures appear without translation or a propulsion wake",
        "trap": "mistaking camera movement or warm terminal features for ferry motion",
        "relation": "fixed slip alignment",
        "secondary": "small harbor craft near the terminal",
    },
    "Ferry Under Way.2.json": {
        "domain": "maritime",
        "target": "the large white ferry underway",
        "setting": "open water with close and wide views",
        "eo": "a broad wake stays aligned behind the ferry",
        "ir": "the same moving vessel and churned-water wake remain visible through contrast changes",
        "trap": "separating the wake from the ferry during wide views",
        "relation": "broad wake following the ferry",
        "secondary": "churned water behind the hull",
    },
    "Ferry.2.json": {
        "domain": "maritime",
        "target": "the white and green ferry",
        "setting": "a coastal channel near shoreline under strong sun glint",
        "eo": "side-on and wider views show the ferry moving near shore",
        "ir": "the hull silhouette, stack region, and long wake persist when EO contrast drops",
        "trap": "letting glare replace the hull as the tracked object",
        "relation": "side silhouette and wake through glare",
        "secondary": "sun glint over the water",
    },
    "Ferry.json": {
        "domain": "maritime",
        "target": "the large ferry",
        "setting": "open water with a smaller faster boat nearby",
        "eo": "the ferry deck and separate wakes are visible from above",
        "ir": "the ferry wake and the smaller boat appear as separate moving objects",
        "trap": "merging the smaller boat and the ferry wake into one object",
        "relation": "separation between the ferry track and the secondary small-boat track",
        "secondary": "the smaller faster boat",
    },
    "Ferry_Under_Way.json": {
        "domain": "maritime",
        "target": "the large double-ended ferry",
        "setting": "open water near island or shoreline background with sun glint",
        "eo": "the hull and wake survive large scale changes and glint intervals",
        "ir": "IR preserves the same hull and wake when EO water contrast is degraded",
        "trap": "treating glare intervals as a target disappearance or replacement",
        "relation": "hull continuity through glint and scale change",
        "secondary": "island or shoreline background",
    },
    "Ferry_v1.json": {
        "domain": "maritime",
        "target": "the white and green ferry near a wooded shoreline",
        "setting": "near-shore water with repeated side views",
        "eo": "side views and scale changes keep the ferry profile stable",
        "ir": "a stable side silhouette and wake distinguish the ferry from the shoreline",
        "trap": "treating visually similar side frames as a stationary vessel",
        "relation": "side silhouette moving with a wake",
        "secondary": "the wooded shoreline",
    },
    "Fishing.Boat.2.json": {
        "domain": "maritime",
        "target": "the small fast motorboat",
        "setting": "open water with close, wide, and glare-affected views",
        "eo": "a long bright wake stays tied to the compact boat through repeated scale changes",
        "ir": "the warm boat and wake relation remain visible when target size changes",
        "trap": "following the wake without keeping the compact boat as the source",
        "relation": "compact boat leading a long wake",
        "secondary": "the long wake",
    },
    "Fishing.Boat.json": {
        "domain": "maritime",
        "target": "the compact fishing boat with visible rigging",
        "setting": "open water with close and wide views",
        "eo": "rigging and a curved or changing wake remain tied to the compact hull",
        "ir": "the warm hull and wake relation persists during the maneuver",
        "trap": "confusing the curved wake with a different path or second vessel",
        "relation": "rigging, compact hull, and curved wake",
        "secondary": "the curved wake",
    },
    "Fishing.boat-3.json": {
        "domain": "maritime",
        "target": "the small motorboat under severe sun glare",
        "setting": "glare-dominated open water with changing contrast",
        "eo": "the boat and wake disappear into glare and then reappear",
        "ir": "the warm compact target and wake remain trackable through degraded visibility",
        "trap": "using the reflection as the target during glare",
        "relation": "boat reappearance after glare",
        "secondary": "saturated reflection on the water",
    },
    "Flatbead.json": {
        "domain": "road",
        "target": "the flatbed semi-truck",
        "setting": "a multi-lane highway through agricultural fields with a concrete overpass",
        "eo": "the truck changes scale and briefly passes under the overpass",
        "ir": "engine and tire heat re-identify the truck after scale and environment changes",
        "trap": "treating the overpass occlusion as a permanent track break",
        "relation": "truck continuity through an overpass occlusion",
        "secondary": "the concrete overpass",
    },
    "Freighter_3000.json": {
        "domain": "maritime",
        "target": "the large cargo vessel or tanker",
        "setting": "high-angle open-water views",
        "eo": "a long hull with colored deck and superstructure stays visible through scale changes",
        "ir": "the same large vessel shape carries localized thermal contrast",
        "trap": "switching from hull geometry to a warm deck patch as the tracked object",
        "relation": "long hull and superstructure continuity",
        "secondary": "localized deck thermal contrast",
    },
    "Garbage.Truck.json": {
        "domain": "road",
        "target": "the automated side-loading garbage truck",
        "setting": "a street collection stop viewed from the air",
        "eo": "the mechanical arm extends to grab a bin while the truck is stopped",
        "ir": "a transient heat burst near the front or cab aligns with hydraulic arm operation",
        "trap": "reading the heat burst as ordinary road background or unrelated engine motion",
        "relation": "arm extension and hydraulic heat timing",
        "secondary": "the collected bin",
    },
    "Helicopter.json": {
        "domain": "aircraft_stationary",
        "target": "the stationary helicopter",
        "setting": "an airport tarmac viewed from a high angle",
        "eo": "main rotor blur is visible while the helicopter does not translate",
        "ir": "a concentrated hotspot beneath the rotor mast marks active engine exhaust",
        "trap": "treating rotor activity as taxiing or relocation",
        "relation": "rotor motion without ground translation",
        "secondary": "the tarmac background",
    },
    "Houseboat.json": {
        "domain": "maritime",
        "target": "the white cabin-style houseboat",
        "setting": "open water with repeated close and wide views",
        "eo": "the boxy cabin and deck details move with a pronounced wake",
        "ir": "the warm cabin structure and disturbed water remain linked",
        "trap": "mistaking the boxy cabin for a stationary dock structure",
        "relation": "cabin cruiser shape and wake",
        "secondary": "disturbed water behind the boat",
    },
    "Large Airplane.json": {
        "domain": "aircraft_stationary",
        "target": "the large four-engine transport aircraft",
        "setting": "an airport apron seen from side profile to top-down orbit",
        "eo": "the aircraft stays static while perspective changes around it",
        "ir": "all four engines remain cold without localized hotspots",
        "trap": "interpreting camera orbit as taxiing or engine activity",
        "relation": "fixed apron position and cold engines",
        "secondary": "apron markings and perspective change",
    },
    "Logging.Truck.json": {
        "domain": "road",
        "target": "the red cab logging truck with a main trailer and pup trailer",
        "setting": "a rural two-lane highway seen from trailing and top-down views",
        "eo": "loaded logs and the multi-part trailer arrangement remain visible",
        "ir": "the hot cab contrasts with cooler logs and sun-warmed road",
        "trap": "dropping the pup trailer or treating the log load as road texture",
        "relation": "cab, main trailer, pup trailer, and load continuity",
        "secondary": "the cooler log payload",
    },
    "Parked.Car.json": {
        "domain": "road_stationary",
        "target": "the red parked car",
        "setting": "an asphalt parking lot with other parked vehicles",
        "eo": "the car remains stationary among similar parked vehicles",
        "ir": "a bright engine-compartment hotspot and a cold moving person near the car appear",
        "trap": "turning residual engine heat or the moving person into vehicle motion",
        "relation": "parked car state with residual heat",
        "secondary": "cold moving person",
    },
    "Passenger.Bus.json": {
        "domain": "road",
        "target": "the passenger bus",
        "setting": "a multi-lane highway with fields, pavement, overpasses, and intersections",
        "eo": "the bus profile stays consistent while it travels along the highway",
        "ir": "thermal contrast changes as the bus crosses cooler fields and warmer pavement",
        "trap": "using thermal contrast change as an identity switch",
        "relation": "bus continuity across changing backgrounds",
        "secondary": "warm pavement and infrastructure",
    },
    "Power.Boat.json": {
        "domain": "maritime",
        "target": "the small powerboat",
        "setting": "open water with glare, close views, and wide views",
        "eo": "the boat changes heading at speed and leaves a long wake",
        "ir": "a warm compact hull and wake remain trackable through the maneuver",
        "trap": "tracking the curved wake as if it were a separate boat",
        "relation": "turning hull and wake geometry",
        "secondary": "the long turning wake",
    },
    "RCN.Ships.1.json": {
        "domain": "maritime_stationary",
        "target": "the cluster of grey ships at the naval base",
        "setting": "piers, cranes, harbor infrastructure, and docked ships",
        "eo": "the camera pans and zooms over moored ships and base structures",
        "ir": "warm dock and ship structures appear without underway motion",
        "trap": "mistaking thermal brightness in the base for a moving ship",
        "relation": "ship cluster fixed to piers",
        "secondary": "cranes and harbor infrastructure",
    },
    "RCN.Ships.2.json": {
        "domain": "maritime_stationary",
        "target": "the single grey naval vessel",
        "setting": "a harbor pier with wide base context and close ship views",
        "eo": "the vessel stays moored along the pier",
        "ir": "the pier-and-ship geometry remains stable without a clear underway wake",
        "trap": "turning the wide-to-close camera change into vessel departure",
        "relation": "moored hull aligned with the pier",
        "secondary": "harbor water and shore thermal structure",
    },
    "SUV.W.Trailer.json": {
        "domain": "road",
        "target": "the dark SUV towing an open utility trailer",
        "setting": "a highway beside an intense sun-heated horizontal thermal band",
        "eo": "the SUV, trailer, and orange payload remain a coupled moving pair",
        "ir": "small cooler moving shapes must be separated from the hot background band",
        "trap": "letting the hot rock-wall band dominate the tracked target",
        "relation": "SUV, trailer, and payload coupling",
        "secondary": "the intense thermal band beside the road",
    },
    "Sailboat.2.json": {
        "domain": "maritime",
        "target": "the sailboat with a prominent mast",
        "setting": "open water with glare, close views, and wide views",
        "eo": "the mast and hull travel with a clear wake while the sail is not the dominant propulsion cue",
        "ir": "hull and wake relation remains visible through glare",
        "trap": "assuming the mast alone explains the motion without checking wake relation",
        "relation": "mast, hull, and wake",
        "secondary": "glare around the sailboat",
    },
    "Sailboat.3.json": {
        "domain": "maritime",
        "target": "the sailboat towing a small dinghy",
        "setting": "strong sun glint with close and wide camera changes",
        "eo": "a small dinghy or tender trails behind the sailboat in close views",
        "ir": "the towed tender and main sailboat remain separable through glint",
        "trap": "merging the tender with the main hull or wake",
        "relation": "main sailboat and trailing tender",
        "secondary": "the towed dinghy",
    },
    "Sailboat.4.json": {
        "domain": "maritime",
        "target": "the small sailboat with a raised white sail",
        "setting": "wide and close water views with a small tender near the stern",
        "eo": "the raised sail and tender relation are visible in close frames",
        "ir": "low-speed motion appears without a high-speed engine-driven wake",
        "trap": "treating the tender or sail reflection as a separate main target",
        "relation": "sail, hull, and tender relation",
        "secondary": "the small tender",
    },
    "Sailboat.json": {
        "domain": "maritime",
        "target": "the single small sailboat with a raised sail",
        "setting": "sun glint, wide shots, and close views",
        "eo": "sail geometry and a mild wake track the same low-speed vessel",
        "ir": "the sail and hull have a low thermal signature without a strong engine plume",
        "trap": "equating glint or a weak thermal return with a powered speedboat",
        "relation": "raised sail and mild wake",
        "secondary": "sun glint near the sailboat",
    },
    "Salt.Barg.json": {
        "domain": "maritime",
        "target": "the large barge or bulk-cargo vessel with a smaller assisting vessel",
        "setting": "open water with a large rectangular cargo form and small craft wakes",
        "eo": "the large rectangular mass and smaller assist craft remain distinct",
        "ir": "the colder large mass separates from sharper thermal or wake cues near the assist vessel",
        "trap": "assigning the small assist-vessel wake to the large barge",
        "relation": "large cargo form plus assist craft",
        "secondary": "the smaller assisting vessel",
    },
    "Sea.Plane.json": {
        "domain": "aircraft_stationary",
        "target": "the red and white Harbour Air seaplane",
        "setting": "a floating seaplane dock with several similar aircraft",
        "eo": "several aircraft appear stationary and similar in status at the docks",
        "ir": "a nose hotspot marks the tracked seaplane while other aircraft are cold-soaked",
        "trap": "transferring the hotspot to a neighboring docked aircraft",
        "relation": "identity of the tracked red and white seaplane",
        "secondary": "nearby cold-soaked seaplanes",
    },
    "Small Aircraft.json": {
        "domain": "aircraft_moving",
        "target": "the small yellow single-engine aircraft",
        "setting": "an airport apron with severe glare and sun-baked concrete",
        "eo": "the plane is initially obscured by glare, then becomes visible as it taxis",
        "ir": "a low-contrast signature and later engine or exhaust hotspot track the active aircraft",
        "trap": "using the early glare interval as evidence that no aircraft is moving",
        "relation": "taxiing aircraft emerging from glare",
        "secondary": "sun-baked concrete and glare",
    },
    "Small.Airplane.2.json": {
        "domain": "aircraft_stationary",
        "target": "the unconventional twin-boom pusher aircraft",
        "setting": "an airport apron with a rear-engine central pod layout",
        "eo": "the twin-boom structure places the engine/propeller at the rear of the fuselage pod",
        "ir": "an intense hotspot appears at the rear of the central pod",
        "trap": "expecting the heat to be at the nose like a standard single-engine aircraft",
        "relation": "rear engine hotspot tied to pusher geometry",
        "secondary": "standard nose-engine expectations",
    },
    "Small.Airplane.json": {
        "domain": "aircraft_moving",
        "target": "the small white high-wing airplane",
        "setting": "airport taxiway or runway surfaces with dark asphalt and lighter concrete",
        "eo": "the airplane taxis with good contrast on asphalt and reduced contrast on concrete",
        "ir": "the cooler aircraft silhouette moves across hot sun-heated pavement",
        "trap": "expecting a bright engine hotspot to be the primary tracking cue",
        "relation": "cool silhouette moving over hot pavement",
        "secondary": "sun-heated pavement",
    },
    "Tanker.json": {
        "domain": "maritime",
        "target": "the large cargo tanker or bulk carrier",
        "setting": "open water with scale changes",
        "eo": "green deck sections, a red or dark hull band, cranes or hatches, and a wake stay tied together",
        "ir": "the same large hull and wake pattern persist through scale changes",
        "trap": "tracking deck structures separately from the hull",
        "relation": "deck, hull, cranes, and wake continuity",
        "secondary": "deck cranes or hatches",
    },
    "Tanker_Horizon.json": {
        "domain": "maritime",
        "target": "the long red tanker or bulk carrier",
        "setting": "strong sun glint with dark silhouette, side views, and wide wake views",
        "eo": "the long hull remains visible through glint and changing viewpoints",
        "ir": "the hull and wake persist despite saturated water regions",
        "trap": "letting saturated water patches replace the long hull track",
        "relation": "long hull and wake through glint",
        "secondary": "saturated water regions",
    },
    "TeraSense.Red.Boat.json": {
        "domain": "maritime",
        "target": "the bright red fast boat",
        "setting": "open water with repeated tight turns and straight high-speed runs",
        "eo": "circular and crescent wake patterns are followed by straight wake segments",
        "ir": "the moving boat and wake match while occasional artifacts remain separate",
        "trap": "treating wake artifacts as a second maneuvering boat",
        "relation": "turning wake changing into straight-run wake",
        "secondary": "occasional IR artifacts",
    },
    "TeraSense.Truck.json": {
        "domain": "road",
        "target": "the white TERRASENSE pickup truck",
        "setting": "an empty parking lot during circular drifting maneuvers",
        "eo": "the truck performs donuts and leaves faint dark tire skid marks",
        "ir": "bright transient tire-heat tracks appear along with the engine hotspot",
        "trap": "treating the hot tire traces as separate moving objects",
        "relation": "truck path and friction-heated tire tracks",
        "secondary": "the hot tire traces left behind",
    },
    "Tug.W.Barges.json": {
        "domain": "maritime",
        "target": "the tug towing two rectangular barges",
        "setting": "open water with close and wide views",
        "eo": "the tug, tow geometry, and two barges remain linked",
        "ir": "the warmer active tug separates from cooler barges and their wake patterns",
        "trap": "treating the barges as self-propelled or detaching the tug",
        "relation": "tug-to-barge tow geometry",
        "secondary": "the cooler barge pair",
    },
    "Utility.Van.json": {
        "domain": "road",
        "target": "the white utility van with a roof rack",
        "setting": "rural roads, open fields, shadows, trees, and railroad tracks",
        "eo": "the white van remains identifiable as it crosses varied rural backgrounds",
        "ir": "engine heat and the van thermal signature anchor the track through shadows or trees",
        "trap": "switching to a similar bright patch or roadside object after a brief obstruction",
        "relation": "van continuity across roads and railroad tracks",
        "secondary": "trees, shadows, and road intersections",
    },
    "Yachts.json": {
        "domain": "maritime",
        "target": "the yacht pair",
        "setting": "open water with two similar motor yachts",
        "eo": "separate hulls and roughly parallel wakes later interact or cross",
        "ir": "two warm hulls and distinct wake trails support non-merging",
        "trap": "collapsing two similar yachts into one identity when wakes cross",
        "relation": "two-hull continuity through interacting wake paths",
        "secondary": "the crossing wake pattern",
    },
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def sent(text: str) -> str:
    text = normalize(text)
    if not text:
        return text
    return text[:1].upper() + text[1:]


def derive_domain(annotation: dict) -> str:
    text = f"{annotation.get('type', '')} {annotation.get('main_event', '')} {annotation.get('event_description', '')}".lower()
    if any(term in text for term in ["helicopter", "aircraft", "airplane", "airliner", "seaplane"]):
        if any(term in text for term in ["taxi", "moving", "active aircraft"]):
            return "aircraft_moving"
        return "aircraft_stationary"
    if any(term in text for term in ["parked", "stationary vehicle"]):
        return "road_stationary"
    if any(term in text for term in ["truck", "van", "bus", "car", "highway", "road"]):
        return "road"
    if any(term in text for term in ["docked", "moored", "terminal", "berth-bound", "naval base"]):
        return "maritime_stationary"
    if any(term in text for term in ["ship", "boat", "ferry", "barge", "tanker", "yacht", "vessel", "sailboat"]):
        return "maritime"
    return "generic"


def fallback_profile(path: Path, annotation: dict) -> dict:
    desc = normalize(annotation.get("event_description", ""))
    first_sentence = desc.split(".", 1)[0].strip() if desc else normalize(annotation.get("main_event", "the target event"))
    return {
        "domain": derive_domain(annotation),
        "target": first_sentence[0].lower() + first_sentence[1:] if first_sentence else "the tracked target",
        "setting": normalize(annotation.get("type", "")).replace("_", " "),
        "eo": first_sentence,
        "ir": "the IR stream provides complementary structure or thermal evidence for the same event",
        "trap": "switching identity during scale, contrast, or viewpoint changes",
        "relation": "target continuity across the sequence",
        "secondary": "nearby clutter or a visually similar distractor",
    }


def profile_for(path: Path, annotation: dict) -> dict:
    profile = fallback_profile(path, annotation)
    profile.update(PROFILE_OVERRIDES.get(path.name, {}))
    for key, value in list(profile.items()):
        if isinstance(value, str):
            profile[key] = normalize(value)
    return profile


def refs_by_uid(annotation: dict) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for question in annotation.get("qa", []):
        uid = str(question.get("uid"))
        refs[uid] = {
            "time_reference_eo": normalize(question.get("time_reference_eo", question.get("time_reference", ""))),
            "time_reference_ir": normalize(question.get("time_reference_ir", question.get("time_reference", ""))),
        }
    return refs


def fill_ref(question: dict, ref: dict[str, str], modality: str) -> None:
    eo = ref.get("time_reference_eo", "")
    ir = ref.get("time_reference_ir", "")
    if modality == "EO":
        question["time_reference_eo"] = eo or ir
        question["time_reference_ir"] = ""
    elif modality == "IR":
        question["time_reference_eo"] = ""
        question["time_reference_ir"] = ir or eo
    else:
        question["time_reference_eo"] = eo or ir
        question["time_reference_ir"] = ir or eo


def make_qa(
    uid: int,
    group_id: str,
    group_family: str,
    group_focus: str,
    group_step: int,
    level: str,
    question: str,
    correct: str,
    distractors: list[tuple[str, str]],
    qtypes: list[str],
    modality: str,
    ref: dict[str, str],
    evidence_note: str,
) -> dict:
    options = {"A": correct}
    roles = {"A": "correct"}
    for label, (role, text) in zip(["B", "C", "D"], distractors):
        options[label] = text
        roles[label] = role
    row = {
        "uid": str(uid),
        "group_id": group_id,
        "group_family": group_family,
        "group_focus": group_focus,
        "group_step": group_step,
        "capability_level": level,
        "question": question,
        "options": options,
        "answer": "A",
        "option_roles": roles,
        "question_type": qtypes,
        "modality_requirement": modality,
        "evidence_note": evidence_note,
    }
    fill_ref(row, ref, modality)
    relabel_question(row, salt=f"{group_id}:{uid}:{group_focus}")
    return row


def relabel_question(question: dict, salt: str) -> None:
    old_options = question["options"]
    old_roles = question["option_roles"]
    old_answer = question["answer"]
    digest = hashlib.sha256(f"{question['uid']}:{salt}".encode()).digest()
    order = [0, 1, 2, 3]
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


def maritime_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    stationary = profile["domain"] == "maritime_stationary"
    if stationary:
        return stationary_maritime_questions(profile, refs)
    return [
        make_qa(1, "C1", "consistency", "spatiotemporal_identity_under_distractors", 1, "L1",
                f"In the early EO span, which relation places {t} in the scene?",
                f"{sent(t)} is tied to {relation} within {setting}.",
                [("relation_shift_error", f"A competing cue around {secondary} becomes the main tracked cue while the original target drops out."),
                 ("fixed_background_error", f"The strongest cue is a fixed shoreline or water patch, not the moving target."),
                 ("single_part_error", f"Only one isolated bright or dark patch is trackable; the hull relation is not stable.")],
                ["spatial_reasoning", "entity_grounding"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "spatiotemporal_identity_under_distractors", 2, "L2",
                "Across the wider EO views, which change should be followed over time?",
                f"The target's scale changes, but the cue for {relation} stays tied to the same moving hull.",
                [("scale_reset_error", f"The scale change marks a new target because the earlier hull cannot be matched."),
                 ("wake_lead_error", f"The wake-like feature should be followed as the leading object ahead of the hull."),
                 ("background_lock_error", f"The safest track is the background texture because the hull is intermittent.")],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", refs["2"], eo),
        make_qa(3, "C1", "consistency", "spatiotemporal_identity_under_distractors", 3, "L2",
                "Which IR pattern matches the same event without changing the tracked object?",
                sent(ir) + ".",
                [("thermal_patch_switch", f"A warm water patch becomes the target while the hull is secondary."),
                 ("secondary_object_switch", f"Thermal evidence from {secondary} is used as the main track instead of the target."),
                 ("wake_only_error", "Only the wake or disturbed water remains useful; the vessel body cannot be matched.")],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", refs["3"], ir),
        make_qa(4, "C1", "consistency", "spatiotemporal_identity_under_distractors", 4, "L2",
                "Which EO observation is the most likely distractor for this sequence?",
                f"{sent(secondary)} can look salient, but that cue should not replace {t} when {relation} remains visible.",
                [("target_underclaim", f"{sent(t)} is too ambiguous to retain once {secondary} is visible."),
                 ("wake_as_target", "The wake itself should be treated as the vehicle-like object throughout the sequence."),
                 ("background_as_target", f"The stable part of {setting} is the object that should be tracked.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO", refs["4"], trap),
        make_qa(5, "C1", "consistency", "cross_modal_event_chain", 5, "L2",
                "In IR, which cue should stay linked with the EO motion pattern?",
                f"{sent(ir)}, while the EO motion remains tied to {relation}.",
                [("modal_mismatch", f"IR points to {secondary}, while EO points to a different moving object."),
                 ("thermal_only_error", "The brightest thermal region alone defines the target even when its geometry does not match."),
                 ("eo_only_error", "The IR stream adds no useful information because water contrast is enough.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["5"], ir),
        make_qa(6, "C1", "consistency", "cross_modal_event_chain", 6, "L2",
                "Which motion account fits the EO camera changes?",
                f"The camera changes scale or angle while {t} keeps a coherent path through {relation}.",
                [("camera_pan_as_motion", "The apparent motion is mainly the camera sweeping over a stationary target."),
                 ("state_break_error", "The target stops being trackable once the viewpoint changes."),
                 ("secondary_takeover", f"A cue around {secondary} becomes the only moving cue after the scale change.")],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", refs["6"], eo),
        make_qa(7, "C1", "consistency", "cross_modal_event_chain", 7, "L3",
                "Which EO/IR pairing describes the same object during the difficult interval?",
                f"EO carries {eo}; IR carries {ir}.",
                [("cross_modal_swap", f"EO follows {t}, but IR should be matched to {secondary}."),
                 ("single_modality_overfit", f"EO alone is enough because IR mainly shows unrelated water or harbor texture."),
                 ("thermal_artifact_match", "A thermal artifact should be paired with the EO hull because it is brighter than the target.")],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "cross_modal_event_chain", 8, "L3",
                "Which account keeps the sequence as one tracked event?",
                f"{sent(t)} remains the tracked event because {relation} links the EO and IR spans.",
                [("identity_switch_error", f"The event is better split when a competing cue becomes visually stronger."),
                 ("background_motion_error", f"The apparent track is mainly {setting} sliding under the camera."),
                 ("wake_only_chain", "The wake or bright water trail is the only consistent object, independent of the hull.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "temporal_relation_and_negative_evidence", 1, "L2",
                "Which EO detail should be checked before accepting a target change?",
                f"Whether {relation} remains connected to {t} across the scale or glare change.",
                [("distractor_priority", f"Whether a competing cue around {secondary} becomes larger or brighter than the target."),
                 ("background_priority", f"Whether {setting} has a stable line or texture near the target."),
                 ("single_frame_priority", "Whether one frame has a sharper outline than the surrounding frames.")],
                ["temporal_reasoning", "evidence_sufficiency"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "temporal_relation_and_negative_evidence", 2, "L2",
                "Which IR detail is most relevant to preserving the event identity?",
                sent(ir) + ".",
                [("brightness_bias", f"The brightest part of {setting} should be treated as the target."),
                 ("secondary_heat_bias", f"{sent(secondary)} should carry the event because it is easier to isolate."),
                 ("wake_detachment_bias", "Wake contrast should be separated from the vessel even when it moves with it.")],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "temporal_relation_and_negative_evidence", 3, "L3",
                "Which sequence account matches both modalities?",
                f"EO and IR both keep {t} tied to {relation}, with {secondary} treated as a competing cue.",
                [("eo_ir_split", f"EO and IR describe different primary objects, so the event should be split by modality."),
                 ("secondary_merge", f"{sent(secondary)} and the target are best merged into one object once the view changes."),
                 ("background_chain", f"The stable background in {setting} is a stronger identity cue than the moving target.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "temporal_relation_and_negative_evidence", 4, "L3",
                "For the full sequence, which interpretation matches the event chain?",
                f"{sent(t)} is tracked through changing visibility while {relation} and IR structure keep the event coherent.",
                [("over_split_event", f"The sequence should be split whenever {secondary} is more visible than the target."),
                 ("over_merge_event", f"All salient water or harbor cues should be merged into the same object track."),
                 ("static_scene_event", f"The event is mainly a static scene inspection of {setting}, not a moving target sequence.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def stationary_maritime_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    return [
        make_qa(1, "C1", "consistency", "fixed_structure_vs_motion", 1, "L1",
                f"In EO, which relation keeps {t} fixed in the scene?",
                f"{sent(t)} stays tied to {relation} within {setting}.",
                [("departure_overclaim", f"The target separates from the terminal or pier and begins a clear transit."),
                 ("secondary_object_switch", f"{sent(secondary)} becomes the primary tracked cue."),
                 ("wake_motion_error", "A broad propulsion wake is the main cue for the target's movement.")],
                ["scene_grounding", "spatial_reasoning"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "fixed_structure_vs_motion", 2, "L2",
                "As the EO view changes scale or angle, what should remain stable?",
                f"The target's placement relative to {relation} stays stable even though the camera view changes.",
                [("camera_motion_as_target_motion", "The changing viewpoint means the target itself has moved away from the berth."),
                 ("secondary_takeover", f"{sent(secondary)} should replace the original target after the wider view."),
                 ("wake_priority", "A water disturbance should outweigh fixed pier or terminal geometry.")],
                ["camera_motion_vs_target_motion", "temporal_reasoning"], "EO", refs["2"], eo),
        make_qa(3, "C1", "consistency", "fixed_structure_vs_motion", 3, "L2",
                "Which IR reading matches the stationary operational state?",
                sent(ir) + ".",
                [("thermal_departure_error", "Thermal activity means the vessel is already underway."),
                 ("cold_shutdown_error", "No onboard or terminal activity can be inferred from IR."),
                 ("modality_split_error", f"IR should be matched to {secondary} rather than the EO target.")],
                ["thermal_evidence_interpretation", "scene_grounding"], "IR", refs["3"], ir),
        make_qa(4, "C1", "consistency", "fixed_structure_vs_motion", 4, "L2",
                "Which cue is a distractor for judging target motion?",
                f"{sent(secondary)} can be visually or thermally salient, but {relation} keeps the target fixed.",
                [("motion_cue", "Translation away from fixed structures is the main cue in the sequence."),
                 ("wake_cue", "A long wake behind the target is the strongest evidence."),
                 ("target_loss", "The target is lost once terminal structures become more visible.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO+IR", refs["4"], trap),
        make_qa(5, "C1", "consistency", "cross_modal_stationary_state", 5, "L2",
                "Which EO/IR combination fits the operational state?",
                f"EO keeps the vessel fixed at {relation}; IR shows activity in the vessel or terminal without proving transit.",
                [("eo_motion_ir_activity", "EO shows departure, and IR activity confirms underway motion."),
                 ("eo_fixed_ir_unrelated", f"EO shows a fixed vessel, but IR mainly describes unrelated {secondary}."),
                 ("eo_uncertain_ir_shutdown", "EO cannot place the vessel, and IR shows a fully cold inactive scene.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["5"], f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "cross_modal_stationary_state", 6, "L2",
                "Which IR feature should not be treated as vessel translation by itself?",
                f"Warm structures, plume-like activity, or water disturbance near {relation} can coexist with a docked or moored state.",
                [("translation_confirmed", "Any warm plume or disturbed water confirms the target has left its fixed position."),
                 ("all_clutter", "All thermal structure should be ignored because EO already shows the scene."),
                 ("new_target", f"The IR feature should be tracked as {secondary} instead of the vessel.")],
                ["thermal_evidence_interpretation", "evidence_sufficiency"], "IR", refs["6"], ir),
        make_qa(7, "C1", "consistency", "cross_modal_stationary_state", 7, "L3",
                "Which cross-modal match preserves the same event?",
                f"EO placement at {relation} and IR activity around the same site describe one stationary operational scene.",
                [("modality_event_split", "EO describes a docked vessel, but IR should be treated as a different underway scene."),
                 ("wake_departure_match", "EO and IR should be matched by a broad wake rather than by fixed structures."),
                 ("secondary_match", f"The primary match should be {secondary}, not the vessel fixed in the EO view.")],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "cross_modal_stationary_state", 8, "L3",
                "Which account keeps the clip coherent through the later views?",
                f"{sent(t)} remains tied to fixed harbor structure; later or warmer cues must be checked before being merged.",
                [("late_merge_error", f"Later views or {secondary} should be merged into the same moving-vessel account."),
                 ("thermal_motion_error", "Warm thermal evidence alone changes the state from docked to underway."),
                 ("eo_only_state", "The IR stream should be ignored because fixed EO geometry is sufficient for all state details.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "stationary_state_chain", 1, "L2",
                "Which EO relation should be preserved when the view widens?",
                f"The vessel's placement relative to {relation}, not the changing screen position.",
                [("screen_position_bias", "The same screen position is the key identity cue."),
                 ("secondary_size_bias", f"The larger or clearer appearance of {secondary} decides the event."),
                 ("wake_bias", "The longest water disturbance decides whether the vessel is underway.")],
                ["temporal_reasoning", "spatial_reasoning"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "stationary_state_chain", 2, "L2",
                "Which IR interpretation fits the fixed-scene evidence?",
                f"{sent(ir)}, but the fixed EO relation still controls whether the vessel is moving.",
                [("hot_equals_transit", "The warmest region directly proves a vessel underway."),
                 ("cold_equals_absent", "Cooler ship areas mean the EO vessel is absent from IR."),
                 ("secondary_heat_track", f"Thermal structure should be assigned to {secondary} as the primary event.")],
                ["thermal_evidence_interpretation", "evidence_sufficiency"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "stationary_state_chain", 3, "L3",
                "Which account combines the EO and IR observations?",
                f"Fixed EO placement and IR activity describe a stationary or berth-bound target with operational heat nearby.",
                [("departure_account", "EO and IR together show a vessel leaving the berth."),
                 ("different_scene_account", f"EO and IR should be assigned to unrelated targets centered on {secondary}."),
                 ("inactive_only_account", "The absence of translation means all thermal activity is irrelevant.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "stationary_state_chain", 4, "L3",
                "For the full clip, which event chain matches the target state?",
                f"{sent(t)} remains fixed to harbor structure while thermal or scene changes are interpreted around that fixed relation.",
                [("underway_chain", "The target begins as docked but should be treated as underway once thermal activity appears."),
                 ("replacement_chain", f"The original target should be replaced by {secondary} in the later part."),
                 ("no_state_chain", "The clip has no stable target-state relation because the camera keeps changing view.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def road_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    stationary = profile["domain"] == "road_stationary"
    if stationary:
        return stationary_road_questions(profile, refs)
    return [
        make_qa(1, "C1", "consistency", "road_identity_and_motion_chain", 1, "L1",
                f"In the early EO span, which relation identifies {t}?",
                f"{sent(t)} is tied to {relation} in {setting}.",
                [("near_vehicle_switch", f"A cue from {secondary} becomes the primary road cue before the target relation is established."),
                 ("roadside_object_error", "A fixed roadside object is the main target cue."),
                 ("single_frame_shape_error", "Only one isolated vehicle outline matters; the route relation is not stable.")],
                ["spatial_reasoning", "entity_grounding"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "road_identity_and_motion_chain", 2, "L2",
                "Across EO scale changes, which continuity should be preserved?",
                f"{sent(relation)} stays coherent even as the camera moves between close and wide views.",
                [("scale_identity_reset", "A scale change should reset identity because the target cannot be compared across views."),
                 ("traffic_takeover", f"{sent(secondary)} should replace the target when that cue is easier to see."),
                 ("background_tracking", f"The most stable part of {setting} should be tracked instead of the vehicle.")],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", refs["2"], eo),
        make_qa(3, "C1", "consistency", "road_identity_and_motion_chain", 3, "L2",
                "Which IR cue should be tied back to the EO track?",
                sent(ir) + ".",
                [("thermal_background_bias", f"The dominant thermal background in {setting} is the target."),
                 ("secondary_heat_bias", f"Thermal evidence from {secondary} is used as the main identity cue."),
                 ("eo_ir_split", "IR shows a different event and should not be matched to the EO road object.")],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", refs["3"], ir),
        make_qa(4, "C1", "consistency", "road_identity_and_motion_chain", 4, "L2",
                "Which potential confusion should be rejected in the EO sequence?",
                f"{sent(trap)} is the main risk; the vehicle relation should be checked across frames.",
                [("fixed_scene_priority", f"The fixed geometry of {setting} is more diagnostic than the moving vehicle."),
                 ("single_frame_priority", "The sharpest single frame is enough to decide identity without continuity."),
                 ("thermal_priority", "EO identity should be ignored once a thermal cue appears later.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO", refs["4"], trap),
        make_qa(5, "C1", "consistency", "cross_modal_road_event", 5, "L2",
                "Which combined observation fits the moving road event?",
                f"EO carries {eo}; IR carries {ir}.",
                [("modal_split", f"EO follows {t}, while IR should be assigned to {secondary}."),
                 ("background_dominance", f"IR background contrast replaces the vehicle track in {setting}."),
                 ("eo_only_chain", "The EO stream alone determines the event, so the IR cue is not part of the answer.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["5"], f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "cross_modal_road_event", 6, "L2",
                "Which motion account separates vehicle motion from camera motion?",
                f"The camera viewpoint changes, but {t} keeps its road relation rather than becoming a fixed scene feature.",
                [("camera_only_motion", "The apparent movement is mainly camera motion over a stationary road object."),
                 ("route_break", "The vehicle path should be treated as broken at each large viewpoint change."),
                 ("traffic_merge", f"The target should merge with {secondary} whenever traffic density increases.")],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", refs["6"], eo),
        make_qa(7, "C1", "consistency", "cross_modal_road_event", 7, "L3",
                "Which EO/IR pairing keeps the same target event?",
                f"{sent(t)} keeps {relation} in EO, and IR provides the corresponding thermal or contrast anchor.",
                [("modal_target_swap", f"EO and IR should be matched to different vehicles, with IR centered on {secondary}."),
                 ("thermal_background_match", f"The thermal background in {setting} is a better match than the vehicle."),
                 ("shape_only_match", "A similar vehicle shape in one frame is enough even when the path relation differs.")],
                ["cross_modal_phenomenon_explanation", "evidence_sufficiency"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "cross_modal_road_event", 8, "L3",
                "Which account matches the full road sequence?",
                f"The target event is carried by {relation}, while {secondary} and background changes are checked as competing cues.",
                [("over_merge", f"{sent(secondary)} should be merged into the same target whenever it appears in a similar road setting."),
                 ("over_split", "Every strong scale change should split the event into a new target."),
                 ("thermal_only", "The IR appearance alone decides identity regardless of the EO path.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "road_negative_evidence_chain", 1, "L2",
                "Which EO check is needed before accepting a vehicle swap?",
                f"Whether {relation} continues across the relevant frames instead of relying on {secondary}.",
                [("appearance_only", "Whether one vehicle outline is sharper than the others."),
                 ("background_only", f"Whether {setting} looks similar across separated views."),
                 ("thermal_only", "Whether a later thermal cue is brighter than the earlier vehicle.")],
                ["temporal_reasoning", "evidence_sufficiency"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "road_negative_evidence_chain", 2, "L2",
                "Which IR detail is useful for road-event continuity?",
                sent(ir) + ".",
                [("hot_background", f"The warmest background feature in {setting} defines the event."),
                 ("secondary_anchor", f"{sent(secondary)} should anchor the event because it is visually plausible."),
                 ("ir_irrelevant", "IR should be ignored whenever EO gives a visible vehicle shape.")],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "road_negative_evidence_chain", 3, "L3",
                "Which sequence account matches the EO and IR evidence?",
                f"{sent(t)} is followed through {relation}; IR is used to check identity through difficult backgrounds.",
                [("different_events", f"EO and IR are better treated as unrelated events centered on {secondary}."),
                 ("fixed_scene", f"The scene geometry of {setting} is the tracked object, not the vehicle."),
                 ("single_frame", "The clearest individual frame should determine the whole sequence.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "road_negative_evidence_chain", 4, "L3",
                "For the full clip, which interpretation follows the event chain?",
                f"{sent(t)} is tracked by its road relation and cross-modal anchor while {trap} is explicitly avoided.",
                [("merge_chain", f"The target should be merged with {secondary} whenever both appear in a road scene."),
                 ("reset_chain", "Each occlusion, zoom, or background change creates a new target event."),
                 ("background_chain", f"The most stable feature of {setting} is the actual tracked target.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def stationary_road_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    return [
        make_qa(1, "C1", "consistency", "stationary_vehicle_state_chain", 1, "L1",
                f"In EO, which relation places {t}?",
                f"{sent(t)} remains parked within {setting}.",
                [("moving_vehicle_error", "The vehicle changes lanes or leaves the lot during the span."),
                 ("person_as_vehicle", f"{sent(secondary)} becomes the tracked vehicle."),
                 ("background_as_vehicle", "A fixed asphalt marking is the main moving object.")],
                ["scene_grounding", "spatial_reasoning"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "stationary_vehicle_state_chain", 2, "L2",
                "Which IR observation changes the state interpretation without implying vehicle motion?",
                sent(ir) + ".",
                [("motion_from_heat", "The hot engine area proves the car is currently driving."),
                 ("person_motion_merge", f"{sent(secondary)} should be merged into the vehicle track."),
                 ("cold_vehicle_error", "No operational information is visible because the car body is stationary.")],
                ["thermal_evidence_interpretation", "evidence_sufficiency"], "IR", refs["2"], ir),
        make_qa(3, "C1", "consistency", "stationary_vehicle_state_chain", 3, "L2",
                "Which moving cue should remain separate from the vehicle state?",
                f"{sent(secondary)} is a separate moving entity, while {t} remains stationary.",
                [("vehicle_motion_error", f"{sent(secondary)} shows the car moving away."),
                 ("engine_as_person", "The engine hotspot should be treated as the moving entity."),
                 ("background_motion", "The asphalt background is moving relative to the parked car.")],
                ["temporal_reasoning", "distractor_rejection"], "IR", refs["3"], trap),
        make_qa(4, "C1", "consistency", "stationary_vehicle_state_chain", 4, "L2",
                "Which EO/IR state pairing describes the car?",
                f"EO shows a parked vehicle; IR adds residual heat and nearby separate motion without turning the car into a moving target.",
                [("moving_car_state", "EO and IR together show the car leaving its parking position."),
                 ("no_operational_state", "EO and IR together show only an inactive cold parked object."),
                 ("person_vehicle_merge", f"The nearby {secondary} should be merged into the car state.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["4"], f"EO: {eo}; IR: {ir}"),
        make_qa(5, "C1", "consistency", "vehicle_state_negative_evidence", 5, "L2",
                "Across EO views, which continuity should be preserved?",
                f"The car's fixed placement in {setting} remains the state cue despite view changes.",
                [("screen_motion", "Screen displacement from camera motion shows the car itself moved."),
                 ("person_priority", f"The path of {secondary} should determine the car's identity."),
                 ("heat_priority", "Thermal state alone should override EO parking geometry.")],
                ["camera_motion_vs_target_motion", "temporal_reasoning"], "EO", refs["5"], eo),
        make_qa(6, "C1", "consistency", "vehicle_state_negative_evidence", 6, "L2",
                "Which alternative should be rejected when judging operation?",
                f"{sent(trap)} should not replace the combined parked-position and heat-state account.",
                [("departure_account", "The vehicle is best described as actively driving during the span."),
                 ("cold_shutdown_account", "The vehicle is best described as cold and inactive with no recent operation cue."),
                 ("new_vehicle_account", "A neighboring parked vehicle should replace the tracked car after the view changes.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO", refs["6"], trap),
        make_qa(7, "C1", "consistency", "vehicle_state_negative_evidence", 7, "L3",
                "Which cross-modal interpretation keeps the target state coherent?",
                f"The stationary EO placement is combined with IR residual heat and a separate nearby moving entity.",
                [("modal_conflict", "EO says parked while IR says a different car is moving, so the event should be split."),
                 ("heat_equals_motion", "The engine hotspot alone is enough to label the car as currently moving."),
                 ("person_merge", f"{sent(secondary)} should be treated as part of the car.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "vehicle_state_negative_evidence", 8, "L3",
                "Which account matches the full parked-vehicle event?",
                f"{sent(t)} remains parked, with IR adding recent-operation evidence and a separate moving distractor.",
                [("active_departure", "The sequence is primarily a moving-car departure event."),
                 ("identity_swap", "The target identity should switch to another object once the person appears."),
                 ("thermal_noise", "The IR evidence is only background noise and should not affect the state.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "parked_state_chain", 1, "L2",
                "Which EO detail should control the motion judgment?",
                f"The car's fixed relation to {setting}, not the apparent screen movement from the camera.",
                [("screen_position", "Screen position alone shows whether the car moved."),
                 ("nearby_person", f"The location of {secondary} controls the car's state."),
                 ("engine_brightness", "Engine brightness alone controls the motion judgment.")],
                ["camera_motion_vs_target_motion", "evidence_sufficiency"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "parked_state_chain", 2, "L2",
                "Which IR detail should be kept separate from vehicle motion?",
                sent(ir) + ".",
                [("motion_from_heat", "The heat pattern shows the vehicle leaving the lot."),
                 ("target_replacement", f"{sent(secondary)} becomes the target in IR."),
                 ("no_state_signal", "The IR pattern provides no state information.")],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "parked_state_chain", 3, "L3",
                "Which sequence account matches both views?",
                f"EO supplies parked geometry; IR supplies residual heat and separates {secondary} from the vehicle.",
                [("moving_vehicle", "Both views show the car transitioning into active travel."),
                 ("unrelated_modalities", "EO and IR describe unrelated parked-lot events."),
                 ("person_identity", f"{sent(secondary)} is the object whose state should be scored.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "parked_state_chain", 4, "L3",
                "For the full clip, which interpretation matches the event chain?",
                f"{sent(t)} is stationary but recently operated; {secondary} remains a separate moving cue.",
                [("driving_chain", "The vehicle is currently driving because IR shows heat."),
                 ("cold_chain", "The vehicle is cold and inactive because EO shows no motion."),
                 ("person_vehicle_chain", f"The vehicle event should be assigned to {secondary}.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def aircraft_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    moving = profile["domain"] == "aircraft_moving"
    if moving:
        return moving_aircraft_questions(profile, refs)
    return [
        make_qa(1, "C1", "consistency", "aircraft_state_under_view_change", 1, "L1",
                f"In EO, which relation places {t}?",
                f"{sent(t)} stays in {setting}; the key relation is {relation}.",
                [("taxi_claim", "The aircraft is leaving its position across the span."),
                 ("neighbor_switch", f"{sent(secondary)} should replace the tracked aircraft."),
                 ("background_track", "The ground markings are the tracked target.")],
                ["scene_grounding", "spatial_reasoning"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "aircraft_state_under_view_change", 2, "L2",
                "Across EO perspective changes, which cue should remain stable?",
                f"The aircraft state is judged from fixed placement and structure, not from camera orbit or screen-side changes.",
                [("orbit_as_taxi", "The orbiting viewpoint shows the aircraft taxiing."),
                 ("screen_side_identity", "The side of the screen is the strongest identity cue."),
                 ("secondary_priority", f"{sent(secondary)} should determine the aircraft state.")],
                ["camera_motion_vs_target_motion", "temporal_reasoning"], "EO", refs["2"], eo),
        make_qa(3, "C1", "consistency", "aircraft_state_under_view_change", 3, "L2",
                "Which IR observation matches the operational state?",
                sent(ir) + ".",
                [("heat_transfer_error", f"The strongest thermal feature belongs to {secondary}, so the target identity changes."),
                 ("motion_from_heat", "Any localized heat means the aircraft is rolling or taking off."),
                 ("eo_only_state", "IR should not affect the aircraft state because EO shows the outline.")],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", refs["3"], ir),
        make_qa(4, "C1", "consistency", "aircraft_state_under_view_change", 4, "L2",
                "Which confusion should be rejected when comparing views?",
                f"{sent(trap)} should not override the aircraft's fixed placement and modality-specific state cues.",
                [("state_from_screen", "Changing screen side is enough to infer aircraft movement."),
                 ("neighbor_aircraft_merge", f"The target should be merged with {secondary}."),
                 ("background_heat_state", "Ground heat alone decides the aircraft's operational state.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO+IR", refs["4"], trap),
        make_qa(5, "C1", "consistency", "aircraft_cross_modal_state", 5, "L2",
                "Which EO/IR combination fits the aircraft event?",
                f"EO carries {eo}; IR carries {ir}.",
                [("eo_ir_split", f"EO follows {t}, while IR should be assigned to {secondary}."),
                 ("movement_overclaim", "EO perspective change and IR heat together show clear taxiing."),
                 ("thermal_omission", "Only the EO structure matters; the IR state cue is unrelated.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["5"], f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "aircraft_cross_modal_state", 6, "L2",
                "Which motion account fits the EO sequence?",
                f"The camera changes viewpoint around {t}, while the aircraft itself keeps its fixed relation in the scene.",
                [("taxi_account", "The aircraft moves through the scene as the camera changes viewpoint."),
                 ("replacement_account", f"{sent(secondary)} becomes the aircraft after the viewpoint shift."),
                 ("ground_motion_account", "The main motion is the ground feature moving as a target.")],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", refs["6"], eo),
        make_qa(7, "C1", "consistency", "aircraft_cross_modal_state", 7, "L3",
                "Which cross-modal account keeps the state coherent?",
                f"EO structure and placement are matched with IR state evidence for the same aircraft.",
                [("modal_conflict", "EO and IR imply different aircraft, so the event should be split."),
                 ("generic_heat", "The strongest heat source in the scene should be assigned to the aircraft regardless of location."),
                 ("neighbor_identity", f"{sent(secondary)} carries the state evidence instead of the target.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "aircraft_cross_modal_state", 8, "L3",
                "Which account matches the full aircraft clip?",
                f"{sent(t)} keeps a stable identity while EO geometry and IR thermal state are interpreted together.",
                [("taxi_overclaim", "The clip is primarily a taxiing sequence driven by camera-orbit cues."),
                 ("identity_switch", f"The target should switch to {secondary} when the view changes."),
                 ("thermal_only", "IR alone determines the state without checking EO structure.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "aircraft_state_chain", 1, "L2",
                "Which EO detail should be checked before inferring aircraft motion?",
                f"Whether the aircraft changes position relative to {setting}, not only how the camera orbit changes the view.",
                [("screen_side", "Whether the aircraft crosses from one screen side to another."),
                 ("neighbor_shape", f"Whether {secondary} has a similar shape."),
                 ("thermal_brightness", "Whether a later thermal cue becomes brighter.")],
                ["camera_motion_vs_target_motion", "evidence_sufficiency"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "aircraft_state_chain", 2, "L2",
                "Which IR cue should be assigned to the target state?",
                sent(ir) + ".",
                [("ground_heat", "Ground or apron heat is the main aircraft state cue."),
                 ("neighbor_heat", f"{sent(secondary)} should receive the target's thermal state."),
                 ("ir_absent", "IR carries no useful state information for the aircraft.")],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "aircraft_state_chain", 3, "L3",
                "Which sequence account matches both modalities?",
                f"EO preserves aircraft identity and placement, while IR resolves the operational state at the same structure.",
                [("different_aircraft", "EO and IR should be assigned to different aircraft because the cues differ."),
                 ("takeoff_chain", "The aircraft transitions toward takeoff because there is thermal structure."),
                 ("background_chain", "The apron or dock structure is the main target in both views.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "aircraft_state_chain", 4, "L3",
                "For the full clip, which interpretation matches the aircraft event?",
                f"{sent(t)} is kept as one target, with motion state judged from fixed placement plus IR thermal evidence.",
                [("motion_from_orbit", "Camera orbit and heat together make the aircraft an active taxiing target."),
                 ("identity_replacement", f"The aircraft identity should be replaced by {secondary}."),
                 ("eo_only", "The EO outline alone is sufficient and the IR state cue should be ignored.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def moving_aircraft_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    t, setting, eo, ir, trap, relation, secondary = fields(profile)
    return [
        make_qa(1, "C1", "consistency", "taxiing_aircraft_visibility_chain", 1, "L1",
                f"In the early EO span, which cue places {t}?",
                f"{sent(t)} is tied to {relation} in {setting}, even when contrast changes.",
                [("glare_as_target", f"{sent(secondary)} becomes the tracked aircraft."),
                 ("static_apron", "A fixed apron marking is the target cue."),
                 ("neighbor_aircraft", "A nearby stationary aircraft carries the event.")],
                ["spatial_reasoning", "entity_grounding"], "EO", refs["1"], eo),
        make_qa(2, "C1", "consistency", "taxiing_aircraft_visibility_chain", 2, "L2",
                "As visibility changes, which EO motion should be followed?",
                f"The aircraft taxis through the scene while {relation} links the difficult and clearer views.",
                [("visibility_reset", "The early low-visibility span and later clearer span should be treated as unrelated."),
                 ("background_motion", "The apparent motion is mainly the camera sweeping over stationary pavement."),
                 ("glare_priority", f"{sent(secondary)} is the most reliable object to track.")],
                ["temporal_reasoning", "camera_motion_vs_target_motion"], "EO", refs["2"], eo),
        make_qa(3, "C1", "consistency", "taxiing_aircraft_visibility_chain", 3, "L2",
                "Which IR cue matches the moving aircraft?",
                sent(ir) + ".",
                [("pavement_heat", f"The heated pavement in {setting} is the aircraft track."),
                 ("neighbor_heat", "A different parked aircraft carries the thermal cue."),
                 ("ir_unusable", "IR provides no way to follow the aircraft through the difficult interval.")],
                ["thermal_evidence_interpretation", "entity_grounding"], "IR", refs["3"], ir),
        make_qa(4, "C1", "consistency", "taxiing_aircraft_visibility_chain", 4, "L2",
                "Which confusion should be rejected when linking the views?",
                f"{sent(trap)} should not replace the moving-aircraft account when later structure and IR cues line up.",
                [("stationary_account", "The aircraft should be treated as stationary because early EO is unclear."),
                 ("background_account", f"The key target is the texture of {setting}."),
                 ("different_aircraft", "The later visible plane is a different aircraft from the early IR cue.")],
                ["distractor_rejection", "evidence_sufficiency"], "EO+IR", refs["4"], trap),
        make_qa(5, "C1", "consistency", "aircraft_motion_cross_modal_chain", 5, "L2",
                "Which EO/IR pairing fits the taxiing event?",
                f"EO carries {eo}; IR carries {ir}.",
                [("modal_split", "EO and IR describe different aircraft because contrast differs."),
                 ("pavement_target", f"The pavement heat in {setting} should be tracked instead of the aircraft."),
                 ("eo_only", "The IR cue is unrelated once the aircraft becomes visible in EO.")],
                ["thermal_evidence_interpretation", "temporal_reasoning"], "EO+IR", refs["5"], f"EO: {eo}; IR: {ir}"),
        make_qa(6, "C1", "consistency", "aircraft_motion_cross_modal_chain", 6, "L2",
                "Which account separates target motion from camera motion?",
                f"The aircraft's own taxiing motion is checked against its path through {setting}, not just against screen drift.",
                [("camera_only", "The aircraft remains fixed and only the camera creates motion."),
                 ("glare_only", f"{sent(secondary)} supplies the apparent movement."),
                 ("neighbor_merge", "Nearby aircraft should be merged into the taxiing path.")],
                ["camera_motion_vs_target_motion", "trajectory_reasoning"], "EO", refs["6"], eo),
        make_qa(7, "C1", "consistency", "aircraft_motion_cross_modal_chain", 7, "L3",
                "Which cross-modal account preserves the moving target?",
                f"EO visibility improves or varies while IR keeps a corresponding aircraft cue through the same path.",
                [("split_by_visibility", "The low-visibility and clear intervals should be separate aircraft events."),
                 ("thermal_background", "The thermal background is the only stable object across the path."),
                 ("neighbor_identity", "A stationary neighboring aircraft carries the identity through the sequence.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["7"], f"EO: {eo}; IR: {ir}"),
        make_qa(8, "C1", "consistency", "aircraft_motion_cross_modal_chain", 8, "L3",
                "Which account matches the full taxiing sequence?",
                f"{sent(t)} remains one moving target, with EO visibility and IR thermal contrast supplying complementary cues.",
                [("stationary_state", "The event is mainly a stationary aircraft inspection."),
                 ("glare_replacement", f"{sent(secondary)} replaces the aircraft as the track."),
                 ("modality_conflict", "EO and IR conflict enough that no single moving target should be kept.")],
                ["cross_modal_phenomenon_explanation", "temporal_reasoning"], "EO+IR", refs["8"], trap),
        make_qa(9, "H1", "coherence", "taxiing_aircraft_chain", 1, "L2",
                "Which EO check helps connect difficult and clearer intervals?",
                f"Whether {relation} carries through the visibility change.",
                [("single_clear_frame", "Whether the clearest frame alone shows the aircraft class."),
                 ("background_match", f"Whether {setting} has similar pavement texture throughout."),
                 ("thermal_brightness", "Whether a later heat source is brighter than the aircraft.")],
                ["temporal_reasoning", "evidence_sufficiency"], "EO", refs["9"], eo),
        make_qa(10, "H1", "coherence", "taxiing_aircraft_chain", 2, "L2",
                "Which IR observation should be matched to the EO aircraft path?",
                sent(ir) + ".",
                [("background_heat", "The pavement heat should replace the aircraft as the target."),
                 ("neighbor_hotspot", "A nearby static aircraft should carry the thermal cue."),
                 ("ir_ignore", "IR should be ignored once EO becomes clear.")],
                ["thermal_evidence_interpretation", "spatial_reasoning"], "IR", refs["10"], ir),
        make_qa(11, "H1", "coherence", "taxiing_aircraft_chain", 3, "L3",
                "Which sequence account matches both streams?",
                f"EO and IR link one aircraft through low visibility, later clearer structure, and continued movement.",
                [("two_aircraft", "The unclear and clear intervals are better treated as two aircraft."),
                 ("static_scene", f"The event is fixed-scene observation of {setting}."),
                 ("glare_event", f"{sent(secondary)} is the event rather than the aircraft.")],
                ["cross_modal_phenomenon_explanation", "causal_reasoning"], "EO+IR", refs["11"], f"EO: {eo}; IR: {ir}"),
        make_qa(12, "H1", "coherence", "taxiing_aircraft_chain", 4, "L3",
                "For the full clip, which interpretation matches the event chain?",
                f"{sent(t)} is tracked as a taxiing aircraft whose visibility and thermal cues vary across time.",
                [("stationary_chain", "The aircraft is stationary and any apparent motion is camera movement."),
                 ("replacement_chain", "The target should be replaced once glare or contrast changes."),
                 ("thermal_background_chain", "The heated pavement is the main tracked object.")],
                ["group_verdict", "temporal_reasoning"], "EO+IR", refs["12"], trap),
    ]


def generic_questions(profile: dict, refs: dict[str, dict[str, str]]) -> list[dict]:
    return road_questions({**profile, "domain": "road"}, refs)


def fields(profile: dict) -> tuple[str, str, str, str, str, str, str]:
    return (
        profile["target"],
        profile["setting"],
        profile["eo"],
        profile["ir"],
        profile["trap"],
        profile["relation"],
        profile["secondary"],
    )


def build_questions(path: Path, annotation: dict) -> list[dict]:
    refs = refs_by_uid(annotation)
    for uid in map(str, range(1, 13)):
        refs.setdefault(uid, {"time_reference_eo": "", "time_reference_ir": ""})
    profile = profile_for(path, annotation)
    domain = profile["domain"]
    if domain.startswith("maritime"):
        rows = maritime_questions(profile, refs)
    elif domain.startswith("road"):
        rows = road_questions(profile, refs)
    elif domain.startswith("aircraft"):
        rows = aircraft_questions(profile, refs)
    else:
        rows = generic_questions(profile, refs)
    for row in rows:
        if row["group_id"] == "C1":
            row["group_focus"] = "v14_relation_chain"
        elif row["group_id"] == "H1":
            row["group_focus"] = "v14_coherence_chain"
    return rows


def validate_text(annotation: dict, path: Path) -> None:
    banned_patterns = [
        r"\bshortcut\b",
        r"single-frame",
        r"category shortcut",
        r"rather than from",
        r"object class only",
        r"benchmark",
    ]
    for question in annotation["qa"]:
        texts = [question["question"], *question["options"].values()]
        blob = "\n".join(texts).lower()
        for pattern in banned_patterns:
            if re.search(pattern, blob):
                raise ValueError(f"{path.name} uid={question['uid']} contains shortcut-prone wording: {pattern}")


def update_notes(annotation: dict) -> None:
    notes = copy.deepcopy(annotation.get("notes", {}))
    notes["design_revision_note"] = (
        "Video-reasoning v14: rebuilt questions around temporal relation tracking, "
        "cross-modal disambiguation, same-scene distractors, and no-video text-only baseline readiness."
    )
    notes["annotation_visibility_warning"] = (
        "Do not export main_event, event_description, answer, option_roles, evidence_note, "
        "rationales, or time_reference metadata to model prompts."
    )
    notes["acceptance_gates"] = [
        "Run prepare-dataset schema validation.",
        "Run request key-path leakage audit after export.",
        "Run Qwen text-only/no-video baseline before video inference.",
        "Reject samples whose text-only accuracy suggests answerable-by-wording shortcuts.",
        "Keep evidence_note fields internal; they are for annotation audit only.",
    ]
    notes["group_scoring_recommendation"] = {
        "C1": "Use squared-ratio consistency scoring for the eight-step relation and cross-modal evidence chain.",
        "H1": "Prefer prefix scoring because the four questions form a coherence chain from temporal check to final event account.",
    }
    annotation["notes"] = notes


def main() -> None:
    changed = []
    for path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        if path.name in PROTECTED:
            continue
        annotation = load_json(path)
        annotation["qa"] = build_questions(path, annotation)
        update_notes(annotation)
        validate_text(annotation, path)
        dump_json(path, annotation)
        changed.append(path.name)
    print(json.dumps({"status": "ok", "changed_count": len(changed), "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
