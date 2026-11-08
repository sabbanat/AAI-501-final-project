"""Court zones for NBA shot-chart coordinates.

x_legacy and y_legacy are tenths of a foot, with the basket at the origin
and positive y toward half court. The three labels match the Kaggle shot
chart: basic zone, side of the floor, and distance band.
"""

import csv
import math
from pathlib import Path


HALF_COURT_FEET = 41.75
CORNER_X_FEET = 22.0
CORNER_Y_FEET = 8.75
ARC_FEET = 23.75
LANE_HALF_WIDTH_TENTHS = 80


def court_zones(x_legacy, y_legacy, shot_value):
    """Return basic zone, side of the floor, and distance band.

    Side labels follow the NBA shot chart. Every attempt inside 8 feet is
    called center, including shots that are not straight on.
    """
    x_tenth = float(x_legacy)
    y_tenth = float(y_legacy)
    x_feet = x_tenth / 10.0
    y_feet = y_tenth / 10.0
    radius = math.hypot(x_feet, y_feet)
    angle = _court_angle(x_feet, y_feet)
    zone_name, zone_range = _side_and_range(x_feet, y_feet, radius, angle)
    basic_zone = _basic_zone(x_tenth, y_tenth, y_feet, radius, shot_value)
    return basic_zone, zone_name, zone_range


def _court_angle(x_feet, y_feet):
    """Degrees from the right sideline, matching the NBA area wedges."""
    if y_feet < 0 and x_feet > 0:
        return 0.0
    if y_feet < 0 and x_feet < 0:
        return 180.0
    if y_feet < 0:
        return 90.0
    return math.degrees(math.atan2(y_feet, x_feet))


def _side_and_range(x_feet, y_feet, radius, angle):
    if y_feet >= HALF_COURT_FEET:
        return "back court", "Back Court Shot"
    if radius <= 8:
        zone_name, zone_range = "center", "Less than 8 ft."
    elif radius <= 16:
        zone_range = "8-16 ft."
        if angle < 60:
            zone_name = "right"
        elif angle <= 120:
            zone_name = "center"
        else:
            zone_name = "left"
    elif radius <= ARC_FEET:
        zone_range = "16-24 ft."
        if angle < 36:
            zone_name = "right"
        elif angle < 72:
            zone_name = "right side center"
        elif angle <= 108:
            zone_name = "center"
        elif angle < 144:
            zone_name = "left side center"
        else:
            zone_name = "left"
    else:
        zone_range = "24+ ft."
        if angle < 72:
            zone_name = "right side center"
        elif angle <= 108:
            zone_name = "center"
        else:
            zone_name = "left side center"

    if abs(x_feet) >= CORNER_X_FEET:
        zone_range = "24+ ft."
        if x_feet > 0 and abs(y_feet) < CORNER_Y_FEET:
            zone_name = "right"
        elif x_feet < 0 and abs(y_feet) < CORNER_Y_FEET:
            zone_name = "left"
        elif x_feet > 0:
            zone_name = "right side center"
        else:
            zone_name = "left side center"
    return zone_name, zone_range


def _basic_zone(x_tenth, y_tenth, y_feet, radius, shot_value):
    if y_feet >= HALF_COURT_FEET:
        return "Backcourt"
    is_three = int(shot_value) == 3
    in_corner = abs(x_tenth) >= 220 and y_tenth <= 87.5
    if is_three and in_corner:
        return "Left Corner 3" if x_tenth < 0 else "Right Corner 3"
    if is_three:
        return "Above the Break"
    if radius < 4:
        return "Restricted Area"
    if radius < 8 and abs(x_tenth) <= LANE_HALF_WIDTH_TENTHS:
        return "In the Paint (non-RA)"
    return "Midrange"


def annotate_file(source, destination):
    """Copy one shot CSV and append basic_zone, zone_name, and zone_range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as infile, destination.open("w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames) + ["basic_zone", "zone_name", "zone_range"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        rows = 0
        for row in reader:
            basic_zone, zone_name, zone_range = court_zones(
                row["x_legacy"], row["y_legacy"], row["shot_value"]
            )
            row["basic_zone"] = basic_zone
            row["zone_name"] = zone_name
            row["zone_range"] = zone_range
            writer.writerow(row)
            rows += 1
    return rows


def _check_known_shots():
    """Published NBA shot-chart rows, coordinates in tenths of a foot."""
    cases = [
        (226, 39, 3, "Right Corner 3", "right", "24+ ft."),
        (-15, 15, 2, "Restricted Area", "center", "Less than 8 ft."),
        (-232, 110, 3, "Above the Break", "left side center", "24+ ft."),
        (-146, 135, 2, "Midrange", "left side center", "16-24 ft."),
        (50, 194, 2, "Midrange", "center", "16-24 ft."),
        (-131, 127, 2, "Midrange", "left side center", "16-24 ft."),
        (-10, 138, 2, "Midrange", "center", "8-16 ft."),
    ]
    for x_legacy, y_legacy, shot_value, basic, side, band in cases:
        got = court_zones(x_legacy, y_legacy, shot_value)
        expected = (basic, side, band)
        if got != expected:
            raise AssertionError(f"{(x_legacy, y_legacy)} -> {got}, expected {expected}")


def main():
    _check_known_shots()
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    sources = sorted(raw_dir.glob("shots_*.csv"))
    if not sources:
        raise SystemExit(f"No shot files in {raw_dir}")
    for source in sources:
        rows = annotate_file(source, out_dir / source.name)
        print(f"{source.name}: {rows} rows -> {out_dir / source.name}")


if __name__ == "__main__":
    main()
