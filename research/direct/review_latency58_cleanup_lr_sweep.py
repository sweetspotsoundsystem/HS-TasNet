"""Record the completed three-rate review from existing quality evidence only."""
from __future__ import annotations

import json
from pathlib import Path

from research.direct.latency58_additional_confirmation_gate import validate_quality_contract, quality_paths
from research.direct.run_latency58_quality import PHASE, ROOT, read, require, sha, write


def main():
    out = PHASE / "cleanup-lr-sweep-review-001"
    require(not out.exists(), "Preserve the completed review")
    evidence = {}

    def bound(path):
        path = Path(path)
        digest = sha(path)
        value = read(path)
        require(sha(path) == digest, "Review input changed while reading")
        evidence[str(path)] = digest
        return value

    manifest = bound(ROOT / "research/manifests/valid.json")
    bundles = {}
    for prefix in ("leader-cleanup-250", "cleanup-lr3e6-250", "cleanup-successor-250", "cleanup-rebound-250"):
        documents = {key: bound(path) for key, path in quality_paths(prefix).items()}
        validate_quality_contract(prefix, documents, {"track_names": [t["name"] for t in manifest["tracks"]]})
        for role in ("primary", "summary"):
            require(documents[role + "_result"]["plan_sha256"]
                    == evidence[str(quality_paths(prefix)[role + "_plan"])], "Quality plan changed")
        bundles[prefix] = documents["summary_result"]

    views = {label: bound(PHASE / (prefix + "-views-001/result.json")) for label, prefix in
             (("accepted", "leader-cleanup-250"), ("lower", "cleanup-lr3e6-250"))}
    for label, prefix in (("accepted", "leader-cleanup-250"), ("lower", "cleanup-lr3e6-250")):
        require(views[label]["status"] == "pass" and views[label]["source_bindings_unchanged"] is True
                and views[label]["model"]["model_state_sha256"] == bundles[prefix]["model_state_sha256"],
                "Controlled views refer to another endpoint")

    window_review = {}
    gains = []
    for view, stem in (("instrumental", "vocals"), ("vocals_only", "other")):
        rows = []
        for before, after in zip(views["accepted"]["tracks"], views["lower"]["tracks"], strict=True):
            require(before["name"] == after["name"], "Track pairing differs")
            for old, new in zip(before["views"][view]["windows"], after["views"][view]["windows"], strict=True):
                require(all(old[key] == new[key] for key in
                            ("physical_start", "physical_end", "input_rms_dbfs", "input_active")),
                        "Physical window or activity support differs")
                if not old["input_active"]:
                    continue
                reference = old["per_stem"][stem]["output_rms_dbfs"]
                candidate = new["per_stem"][stem]["output_rms_dbfs"]
                rows.append({"track": before["name"], "physical_start_seconds": old["physical_start"] / 44100,
                             "reference_dbfs": reference, "candidate_dbfs": candidate, "delta_db": candidate - reference})
                if view == "vocals_only":
                    old_gain = old["per_stem"]["vocals"]["signed_desired_projection_gain"]
                    new_gain = new["per_stem"]["vocals"]["signed_desired_projection_gain"]
                    gains.append(abs(1 - new_gain) - abs(1 - old_gain))
        window_review[view + "_to_" + stem] = {
            "eligible_windows": len(rows), "lower_output_windows": sum(r["delta_db"] < 0 for r in rows),
            "higher_output_windows": sum(r["delta_db"] > 0 for r in rows),
            "worst_three_output_increases": sorted(rows, key=lambda r: r["delta_db"], reverse=True)[:3]}
    require(len(gains) == 319 and window_review["instrumental_to_vocals"]["eligible_windows"] == 420,
            "Controlled support changed")
    window_review["wanted_vocal_gain_error"] = {
        "eligible_windows": len(gains), "improved_windows": sum(d < 0 for d in gains),
        "window_weighted_mean_absolute_gain_error_delta": sum(gains) / len(gains),
        "aggregation": "Each input-active one-second window receives equal weight; not a track bootstrap."}

    rates = {}
    for label, prefix, rate in (("lower", "cleanup-lr3e6-250", 3e-6),
                               ("middle", "cleanup-successor-250", 1e-5),
                               ("higher", "cleanup-rebound-250", 3e-5)):
        summary = bundles[prefix]
        rates[label] = {"prefix": prefix, "maximum_learning_rate": rate,
                        "model_state_sha256": summary["model_state_sha256"],
                        "full_mixture_aggregate": summary["full_mixture_aggregate"],
                        "versus_accepted": summary["comparisons"]["parent"]["full14"],
                        "controlled_views_versus_accepted": summary["comparisons"]["parent"]["vocal_views"]}

    lower = bundles["cleanup-lr3e6-250"]
    audit = bound(PHASE / "cleanup-lr3e6-to-000250-001/audit.json")
    require(audit["status"] == "pass" and audit["step"] == 250
            and audit["matched_input_journal_verified"] is True
            and audit["source_bindings_unchanged"] is True
            and lower["matched_input_teacher_rng_updates"] == 250
            and lower["comparison_variable"] == "matched_learning_rate_schedule", "Matched-rate audit incomplete")

    from research.direct.train_latency58 import disk_bytes
    roots = [PHASE.parent / "latency11", PHASE, ROOT.parent / "stemgen-rt-cropped1024-11ms",
             ROOT.parent / "stemgen-rt-hop128-5ms"]
    counted = sum(disk_bytes(path) for path in roots)
    outside = 146342157 + 111344465 + disk_bytes(ROOT / ".git/lfs") + disk_bytes(ROOT / ".git/objects")
    require(outside < 500_000_000 and counted + 352_000_000 < 79_500_000_000
            and counted + outside + 352_000_000 < 80_000_000_000,
            "Preserve remaining quarter-trial checkpoint and report space")
    for path in (Path(__file__), ROOT / "research/direct/latency58_additional_confirmation_gate.py",
                 ROOT / "research/direct/run_latency58_quality.py"):
        evidence[str(path)] = sha(path)
    require(all(sha(path) == digest for path, digest in evidence.items()), "Review evidence changed")
    result = {
        "schema": "latency58-cleanup-learning-rate-review-v1", "status": "completed_review",
        "source_bindings": evidence, "reviewed_documents_unchanged": True,
        "input_verification_scope": "Rehashed the listed metadata and review code; original execution receipts authenticate scoring. No audio rehash or decoding in this review.",
        "rates": rates, "lower_rate_window_review": window_review,
        "review": {
            "full_band_sdr": "Lower has the highest three-rate point estimate, 4.061636 dB, still 0.007443 below accepted. Its paired interval versus accepted [-0.023400,+0.004738] includes zero, as do comparisons with the two other rates. This establishes neither superiority nor equivalence.",
            "low_band_fidelity": "Lower minus accepted 20–250 Hz SDR is -0.010527 dB. Skelpolu drums lose 0.261878 dB and 20–80 Hz bass loses 0.308000 dB. All subband and track/stem cells remain in the bound full summary.",
            "per_stem": "Lower full-SDR deltas for drums/bass/vocals/Other are -0.024684/-0.011713/+0.005096/+0.001530 dB. Skelpolu drums lose 0.240007 dB. Controlled wanted instrumental drums/bass/Other also lose 0.019704/0.005417/0.005072 dB.",
            "interference": "Lower minus accepted SIR is -0.014876 dB with a zero-spanning interval; drum SIR loses 0.099499 dB and Skelpolu drums lose 0.335800 dB. All four probes and stems were reviewed: bass unexplained/output changes on notes/hop/harmonic stimuli are +0.029450/-0.071702/+0.041844 dB, while native bass levels change -0.012583/+0.006860/-0.015369 dB. Probe changes are mixed and do not establish audibility.",
            "absence": "Lower minus accepted native absent-output levels for drums/bass/vocals/Other are +0.101214/-0.149151/+0.026587/-0.133894 dB. ANiMAL's one absent-drum window increases 0.390760 dB. Silence-probe vocal output increases 0.153426 dB to -111.504112 dBFS; other stems decrease. Sparse natural absence and synthetic silence are different measurements.",
            "controlled_views": "Lower vocal-only Other output falls 0.211935 dB and wanted vocal SDR rises 0.100873 dB versus accepted, on all 14 track means. Instrumental-input Vocal output rises 0.010238 dB with interval [-0.016923,+0.040905], with 210/420 windows improving. Skelpolu vocal-only SDR remains 0.566059 dB and desired gain 0.081128. Lower preserves somewhat more instrumental fidelity than the middle/higher rates but achieves less vocal-only cleanup. There is no clear overall winner for the two requested bleed directions.",
            "actions60": "Lower full/low/SIR improve +0.048470/+0.056793/+0.010712 dB versus accepted on the one Actions excerpt. Vocal full SDR falls 0.000544 dB and drum SIR falls 0.079911 dB. This diagnostic cannot override the full panel and gives no track-sampling uncertainty estimate.",
            "listening_limitations": "No human comparison of these three rates has been supplied. Existing accepted-model M4 practical adoption does not establish a listening verdict for a new checkpoint. Full and controlled objective results do not establish audible superiority."
        },
        "decision": "Close the three matched learning-rate schedules at their completed 250-update limits without selecting a replacement from the sweep. Retain the accepted model while the separately planned quarter-frequency trial finishes.",
        "decision_basis": "The small vocal-only benefit comes with no established improvement in instrumental-input Vocal spill, slightly weaker wanted instruments, and retained local failures. The small full-SDR decrease alone is not the selection rule.",
        "quality_selected": False, "new_training_updates": 0, "new_model_inference": False,
        "audio_decoded": False, "audio_exported": False, "additional_confirmation_consumed": False,
        "human_listening_completed": False, "goal_complete": False,
        "counted_bytes": counted, "outside_bytes": outside, "reserved_checkpoint_and_reports_bytes": 352_000_000,
        "limitations": ["One seed and the same 4000 augmented examples; nominal track intervals omit seed uncertainty and repeated selection.",
                        "The frozen additional confirmation material is still unused; the final candidate pool review remains pending.",
                        "Existing paced-timing failures and the working M4 rollback remain unchanged."]}
    out.mkdir()
    write(out / "result.json", result)
    require(read(out / "result.json") == result, "Review serialization differs")
    print(json.dumps({"status": result["status"], "path": str(out / "result.json"),
                      "sha256": sha(out / "result.json"), "reviewed_metadata_files": len(evidence)}))


if __name__ == "__main__":
    main()
