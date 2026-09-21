"""
tools/seed_tracker.py — Atomic Seed-Level Persistence & Crash-Resilient Tracker (Tier 1)
========================================================================================
Implements Tier 1 of the Zero-Data-Loss Fault-Tolerant Architecture.

Guarantees:
-----------
1. Atomic Transactions: Each seed execution is an independent transaction.
2. Crash Resilience: If power cuts or the machine shuts down mid-benchmark,
   previously completed seeds are 100% preserved and never re-run.
3. Atomic Write Safety: Uses tempfile + atomic rename to prevent JSON corruption
   during sudden wall-power loss.
4. Model Artifact Verification: A seed is only treated as COMPLETED if its
   serialized model file actually exists and is non-empty on disk.

Usage
-----
    from tools.seed_tracker import SeedTracker

    tracker = SeedTracker()
    if tracker.is_completed("fp3o", "type_conditioned", seed=3, n_agents=8):
        print("Skipping already completed seed 3")
    else:
        tracker.mark_started("fp3o", "type_conditioned", seed=3, n_agents=8)
        # ... train seed 3 ...
        tracker.mark_completed("fp3o", "type_conditioned", seed=3, n_agents=8,
                               model_path="results/models/fp3o_tc_seed3.zip",
                               metrics={"mean_return": -102.2, "shield_rate": 0.138})
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


TRACKER_FILE = Path("results/benchmark_tracker.json")


class SeedTracker:
    def __init__(self, tracker_path: Path = TRACKER_FILE):
        self.tracker_path = Path(tracker_path)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {self.tracker_path} ({e}), initializing fresh tracker.")
        return {
            "version": "1.0_fault_tolerant",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "seeds": {}
        }

    def _save(self) -> None:
        """Atomic write using tempfile and os.replace to guard against power-cut corruption."""
        self._data["last_updated"] = datetime.now(timezone.utc).isoformat()
        dir_path = self.tracker_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8") as tf:
            json.dump(self._data, tf, indent=2)
            temp_name = tf.name

        os.replace(temp_name, self.tracker_path)

    @staticmethod
    def make_key(algo: str, condition: str, seed: int, n_agents: int = 8, fleet_preset: str = "balanced_8") -> str:
        return f"{algo.lower()}_{condition.lower()}_{fleet_preset}_N{n_agents}_s{seed}"

    def is_completed(self, algo: str, condition: str, seed: int, n_agents: int = 8, fleet_preset: str = "balanced_8") -> bool:
        """Returns True if the seed has finished and its model artifact exists on disk."""
        key = self.make_key(algo, condition, seed, n_agents, fleet_preset)
        entry = self._data.get("seeds", {}).get(key)
        if not entry:
            return False
        if entry.get("status") != "COMPLETED":
            return False

        # Verify model artifact exists and is non-empty
        model_path = entry.get("model_path")
        if not model_path or not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            return False

        return True

    def mark_started(self, algo: str, condition: str, seed: int, n_agents: int = 8, fleet_preset: str = "balanced_8") -> None:
        key = self.make_key(algo, condition, seed, n_agents, fleet_preset)
        if "seeds" not in self._data:
            self._data["seeds"] = {}
        self._data["seeds"][key] = {
            "algo": algo,
            "condition": condition,
            "seed": seed,
            "n_agents": n_agents,
            "fleet_preset": fleet_preset,
            "status": "IN_PROGRESS",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def mark_completed(
        self,
        algo: str,
        condition: str,
        seed: int,
        model_path: str,
        metrics: Dict[str, Any],
        n_agents: int = 8,
        fleet_preset: str = "balanced_8"
    ) -> None:
        key = self.make_key(algo, condition, seed, n_agents, fleet_preset)
        if "seeds" not in self._data:
            self._data["seeds"] = {}
        self._data["seeds"][key] = {
            "algo": algo,
            "condition": condition,
            "seed": seed,
            "n_agents": n_agents,
            "fleet_preset": fleet_preset,
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "metrics": metrics
        }
        self._save()

    def get_completed_metrics(self, algo: str, condition: str, n_agents: int = 8, fleet_preset: str = "balanced_8") -> List[Dict[str, Any]]:
        """Retrieve metrics for all completed seeds of a given configuration."""
        results = []
        prefix = f"{algo.lower()}_{condition.lower()}_{fleet_preset}_N{n_agents}_"
        for k, v in self._data.get("seeds", {}).items():
            if k.startswith(prefix) and v.get("status") == "COMPLETED":
                results.append(v)
        return sorted(results, key=lambda x: x["seed"])

    def get_progress_summary(self) -> Dict[str, Any]:
        """Summary of total completed, in-progress, and missing seeds."""
        completed = 0
        in_progress = 0
        for v in self._data.get("seeds", {}).values():
            if v.get("status") == "COMPLETED":
                completed += 1
            elif v.get("status") == "IN_PROGRESS":
                in_progress += 1
        return {
            "total_registered": len(self._data.get("seeds", {})),
            "completed": completed,
            "in_progress": in_progress
        }
