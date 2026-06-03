"""
Metrics Collection and Reporting
Collects, aggregates, and exports benchmark results.
"""
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class MetricsCollector:
    """Collects and exports benchmark metrics."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self, results: dict, formats: Optional[list] = None):
        """Save results in multiple formats."""
        if formats is None:
            formats = ["json", "csv"]

        if "json" in formats:
            self._save_json(results)
        if "csv" in formats:
            self._save_csv(results)

    def _save_json(self, results: dict):
        """Save results as JSON."""
        path = self.output_dir / f"benchmark_{self.timestamp}.json"
        metadata = {
            "timestamp": self.timestamp,
            "version": "1.0.0",
            "results": results,
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Results saved to {path}")

    def _save_csv(self, results: dict):
        """Save results as flattened CSV."""
        path = self.output_dir / f"benchmark_{self.timestamp}.csv"
        rows = []
        self._flatten(results, [], rows)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "metric", "value", "unit"])
            writer.writerows(rows)
        print(f"  Results saved to {path}")

    def _flatten(self, data: Any, prefix: list, rows: list):
        """Recursively flatten nested dicts into rows."""
        if isinstance(data, dict):
            if "value" in data and "unit" in data:
                rows.append([
                    prefix[0] if prefix else "",
                    " / ".join(prefix[1:]) if len(prefix) > 1 else "",
                    data["value"],
                    data["unit"],
                ])
            else:
                for key, value in data.items():
                    self._flatten(value, prefix + [key], rows)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._flatten(item, prefix + [str(i)], rows)

    def load(self, path: str) -> dict:
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return data.get("results", data)

    def compare(self, baseline: dict, current: dict) -> dict:
        """Compare two result sets and compute deltas."""
        comparison = {}
        for category in baseline:
            if category not in current:
                continue
            comparison[category] = self._compare_dict(
                baseline[category], current[category], category
            )
        return comparison

    def _compare_dict(self, baseline: dict, current: dict, path: str) -> dict:
        """Compare two nested dicts."""
        result = {}
        for key in baseline:
            if key not in current:
                continue
            b_val = baseline[key]
            c_val = current[key]
            if isinstance(b_val, dict) and "value" in b_val and isinstance(c_val, dict) and "value" in c_val:
                b_num = float(b_val["value"])
                c_num = float(c_val["value"])
                delta = c_num - b_num
                pct = (delta / b_num * 100) if b_num != 0 else 0
                result[key] = {
                    "baseline": b_val["value"],
                    "current": c_val["value"],
                    "delta": round(delta, 2),
                    "delta_pct": round(pct, 1),
                    "unit": c_val.get("unit", ""),
                }
            elif isinstance(b_val, dict) and isinstance(c_val, dict):
                result[key] = self._compare_dict(b_val, c_val, f"{path}.{key}")
        return result

    def print_summary(self, results: dict):
        """Print a concise text summary."""
        print(f"\n{'=' * 60}")
        print(f"  GPU Benchmark Summary — {self.timestamp}")
        print(f"{'=' * 60}")
        for category, data in results.items():
            if isinstance(data, dict):
                for metric, value in data.items():
                    if isinstance(value, dict) and "value" in value:
                        print(f"  {category}/{metric}: {value['value']} {value.get('unit', '')}")
        print(f"{'=' * 60}\n")
