"""Measure full snapshot construction and publication around the frozen test."""
from __future__ import annotations

import time
from unittest.mock import patch

from research.direct import check_latency58_grouped_vocal_recovery as checked
from research.direct.run_latency58_quality import require


def exercise_recovery(parent, source, out):
    constructions, publications = {}, []
    original_make, original_publish = checked.make_snapshot, checked.publish_snapshot

    def make(*args, **kwargs):
        began = time.monotonic()
        result = original_make(*args, **kwargs)
        constructions[result["step"]] = time.monotonic() - began
        return result

    def publish(snapshot, *args, **kwargs):
        began = time.monotonic()
        result = original_publish(snapshot, *args, **kwargs)
        publications.append({"step": snapshot["step"], "construction_seconds": constructions[snapshot["step"]],
            "publication_seconds": time.monotonic() - began})
        return result

    with patch.object(checked, "make_snapshot", make), patch.object(checked, "publish_snapshot", publish):
        result = checked.exercise_recovery(parent, source, out)
    require([row["step"] for row in publications] == [2, 3], "Wrong successful publication inventory")
    result["complete_successful_save_seconds"] = [row["construction_seconds"] + row["publication_seconds"]
                                                for row in publications]
    result["complete_save_timing"] = publications
    require(max(result["complete_successful_save_seconds"]) < 10,
            "Full recovery save exceeds the margin retained inside the existing progress guard")
    return result
