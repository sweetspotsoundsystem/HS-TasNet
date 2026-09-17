import json
import sys
from pathlib import Path
from research.direct.run_latency58_quality import read, require, sha, write

plan_path = Path(sys.argv[1])
require(sha(plan_path) == sys.argv[2], "Plan changed")
plan = read(plan_path)
require(all(sha(p) == s for p, s in plan["source_bindings"].items()), "Input changed")
import torch
from research import experiment
from research.direct.train_latency58 import PRODUCTION
from research.direct.check_latency58_sdr_context_data import digest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
sys.path.insert(0, str(PRODUCTION))
import train_production as production
config = read(PRODUCTION / "full_config.json")
_, tracks, _, _ = production.load_corpus_manifest(PRODUCTION / "manifests/combined.manifest.json",
    expected_file_sha256=plan["manifest_sha256"], config=config)
dataset = production.CounterAddressedCropDataset(tracks, root_weights=config["sampling"]["root_weights"],
    seed=config["seed"], crop_samples=176128, vocal_active_probability=config["sampling"]["vocal_active_probability"],
    final_sample_index=920004)
batch = [dataset[i] for i in range(920000, 920004)]
x, y = (torch.stack([row[k] for row in batch]) for k in (0, 1))
original_sha = digest((("mixture", x), ("targets", y)))
rows = []
for seed in (3, 1):
    torch.manual_seed(seed)
    rng = torch.get_rng_state().clone()
    draws = torch.rand(4)
    subset = draws < experiment.STEM_SUBSET_AUGMENT_PROBABILITY
    deranged = (draws >= experiment.STEM_SUBSET_AUGMENT_PROBABILITY) & (
        draws < experiment.STEM_SUBSET_AUGMENT_PROBABILITY + experiment.VOCAL_DERANGEMENT_PROBABILITY)
    torch.set_rng_state(rng)
    full = experiment._augment_training_distribution(mixture=x, targets=y)
    parts = []
    for region in (slice(0, 88064), slice(88064, None)):
        torch.set_rng_state(rng)
        parts.append(experiment._augment_training_distribution(mixture=x[..., region], targets=y[..., region]))
    require(torch.equal(full[2], deranged) and torch.equal(parts[0][2], deranged)
            and torch.equal(parts[1][2], deranged), "Vocal-shuffle flags differ across the physical boundary")
    require(all(torch.equal(full[k], torch.cat((parts[0][k], parts[1][k]), dim=-1)) for k in (0, 1)),
            "Prefix and suffix use inconsistent source masks or donors")
    require(torch.equal(full[1][deranged, 2], y[:, 2].roll(shifts=1, dims=0)[deranged]),
            "Vocal donor changes across the four-second crop")
    rows.append({"seed": seed, "subset_examples": int(subset.sum()), "deranged_examples": int(deranged.sum()),
        "original_examples": int((~(subset | deranged)).sum()), "prefix_suffix_exact": True,
        "vocal_donor_continuous": True, "augmented_sha256_cpu": digest(zip(("mixture", "targets", "deranged"), full))})
require(all(sum(r[k] for r in rows) > 0 for k in ("subset_examples", "deranged_examples", "original_examples"))
        and original_sha == digest((("mixture", x), ("targets", y))) and not torch.cuda.is_initialized()
        and all(sha(p) == s for p, s in plan["source_bindings"].items()), "Coverage or input preservation failed")
write(plan_path.parent / "result.json", {"schema": "latency58-context-augmentation-proof-v1", "status": "pass",
    "plan_sha256": sys.argv[2], "source_bindings": plan["source_bindings"], "source_bindings_unchanged": True,
    "cases": rows, "input_sha256": original_sha, "sample_indices": list(range(920000, 920004)),
    "model_loaded": False, "cuda_initialized": False, "scope": "CPU branch coverage; no GPU RNG equivalence claim"})
print(json.dumps({"status": "pass", "cases": rows}))
