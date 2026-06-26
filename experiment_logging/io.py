import csv
import json
import os


def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

