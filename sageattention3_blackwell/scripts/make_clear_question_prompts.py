#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


QUESTION_MARKER = "QUESTION SECTION\n"
ANSWER_MARKER = "ANSWER SECTION\n"


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    summary: str
    inspectors: tuple[str, ...]
    checkpoints: tuple[str, ...]
    materials: tuple[str, ...]
    rules: tuple[str, ...]
    place_word: str
    destination_word: str
    count_word: str
    base_count: int
    backup_delta: int


SCENARIOS = (
    Scenario(
        slug="logistics",
        title="CLEAR LOGISTICS REPORT",
        summary=(
            "The report describes a calm logistics drill for a coastal emergency depot. "
            "Every record uses the same structure so that the facts are easy to follow: "
            "time, inspector, checkpoint, material, bay, staging lane, rule, measured count, and backup count. "
            "The final question asks for details that appear in the report, not for outside knowledge."
        ),
        inspectors=("Jon", "Leah", "Omar", "Nia", "Mira"),
        checkpoints=("East Warehouse", "River Gate", "Hill Office", "South Yard", "North Pier"),
        materials=("labeled battery packs", "blue medical crates", "folded shelter panels", "green radio kits", "sealed water filters"),
        rules=(
            "photograph the pallet label after unloading",
            "keep cold-storage items under the white canopy",
            "send a radio update before the truck leaves",
            "record damaged seals in the shared notebook",
            "verify the manifest before loading any crate",
        ),
        place_word="bay",
        destination_word="staging lane",
        count_word="units",
        base_count=120,
        backup_delta=2,
    ),
    Scenario(
        slug="clinic",
        title="CLEAR MOBILE CLINIC REPORT",
        summary=(
            "The report describes a supervised mobile clinic rehearsal after a regional storm. "
            "Every record uses the same structure so that the facts are easy to follow: "
            "time, coordinator, clinic station, supply type, cabinet, service desk, rule, checked count, and reserve count. "
            "The final question asks for details that appear in the report, not for outside knowledge."
        ),
        inspectors=("Asha", "Ben", "Clara", "Dev", "Elena"),
        checkpoints=("Triage Tent", "Pharmacy Desk", "Vaccine Room", "Intake Hall", "Supply Van"),
        materials=("orange wound-care boxes", "sealed insulin coolers", "paper intake packets", "portable lamp cases", "sterile glove cartons"),
        rules=(
            "initial every temperature label before release",
            "keep controlled medicine inside the locked cart",
            "scan each patient packet before filing it",
            "test every lamp battery before the evening shift",
            "separate torn cartons from clean inventory",
        ),
        place_word="cabinet",
        destination_word="service desk",
        count_word="items",
        base_count=210,
        backup_delta=3,
    ),
    Scenario(
        slug="observatory",
        title="CLEAR OBSERVATORY REPORT",
        summary=(
            "The report describes a quiet instrument calibration run at a mountain observatory. "
            "Every record uses the same structure so that the facts are easy to follow: "
            "time, technician, telescope station, equipment type, storage shelf, calibration bench, rule, measured count, and reserve count. "
            "The final question asks for details that appear in the report, not for outside knowledge."
        ),
        inspectors=("Iris", "Mateo", "Rin", "Selma", "Tariq"),
        checkpoints=("Mirror Lab", "North Dome", "Spectrograph Room", "Control Annex", "Weather Deck"),
        materials=("silver alignment caps", "violet sensor trays", "coded fiber bundles", "black filter wheels", "wrapped tripod heads"),
        rules=(
            "log the humidity reading before opening the case",
            "cover exposed lenses whenever the dome door moves",
            "compare serial numbers against the blue worksheet",
            "save the calibration trace before changing filters",
            "tag any scratched connector for optical review",
        ),
        place_word="storage shelf",
        destination_word="calibration bench",
        count_word="pieces",
        base_count=310,
        backup_delta=4,
    ),
)


TIMES = ("08:10", "10:30", "13:20", "16:50", "06:40")


def record_values(scenario: Scenario, record_id: int) -> dict[str, object]:
    idx = (record_id - 1) % 5
    return {
        "time": TIMES[idx],
        "inspector": scenario.inspectors[idx],
        "checkpoint": scenario.checkpoints[idx],
        "material": scenario.materials[idx],
        "place": (record_id * 7) % 19 + 1,
        "destination": (record_id * 3) % 11 + 1,
        "rule": scenario.rules[idx],
        "measured": scenario.base_count + record_id,
        "backup": scenario.base_count + record_id - scenario.backup_delta,
    }


def make_prompt(scenario: Scenario, records: int = 58) -> str:
    lines = [
        scenario.title,
        "",
        f"Summary: {scenario.summary}",
        "",
    ]
    for record_id in range(1, records + 1):
        values = record_values(scenario, record_id)
        lines.append(
            f"Record {record_id:03d}: At {values['time']}, {values['inspector']} inspected the "
            f"{values['checkpoint']} checkpoint. The team moved {values['material']} from "
            f"{scenario.place_word} {values['place']} to {scenario.destination_word} {values['destination']}. "
            f"The standing rule was to {values['rule']}. The measured count was {values['measured']} "
            f"{scenario.count_word}, and the backup count was {values['backup']} {scenario.count_word}."
        )
        lines.append("")

    first = record_values(scenario, 42)
    second = record_values(scenario, 57)
    diff = int(first["measured"]) - int(first["backup"])
    lines.extend(
        [
            QUESTION_MARKER.rstrip(),
            (
                f"Answer using only the report above. Which inspector handled Record 042, which checkpoint did that "
                f"record mention, and what material was moved? What standing rule applied to Record 042? "
                f"Also compare the measured count and backup count in Record 042, and state the difference "
                f"in {scenario.count_word}. Finally, name the time and {scenario.destination_word} "
                f"from Record 057."
            ),
            "",
            ANSWER_MARKER.rstrip(),
            (
                f"Record 042 was handled by {first['inspector']} at the {first['checkpoint']} checkpoint. "
                f"The moved material was {first['material']}. The standing rule was to {first['rule']}. "
                f"The measured count was {first['measured']} {scenario.count_word}, the backup count was "
                f"{first['backup']} {scenario.count_word}, and the difference was {diff} {scenario.count_word}. "
                f"Record 057 used time {second['time']} and {scenario.destination_word} {second['destination']}."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("tests/prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        path = out_dir / f"clear_question_{scenario.slug}.txt"
        path.write_text(make_prompt(scenario))
        print(path)


if __name__ == "__main__":
    main()
