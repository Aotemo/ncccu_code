"""Extract retained checkpoint loading parameters on a verified compute Pod CPU."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone


def check_cpu_environment():
    if socket.gethostname() != os.environ.get("NCCCU_POD_NAME") or not os.environ.get("NCCCU_POD_UID"):
        raise RuntimeError("Run through the verified Pod CPU transport")
    if os.path.realpath(sys.executable) != os.path.realpath(os.environ["NCCCU_PYTHON"]):
        raise RuntimeError("Discovered Python differs")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("CPU work requires nice 19")
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(key) != "":
            raise RuntimeError("CPU work requires hidden accelerators")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "2":
            raise RuntimeError("CPU work requires at most two threads")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Deployed canonical repository")
    parser.add_argument("--output", type=Path, required=True, help="New technical-prep output directory")
    args = parser.parse_args()
    check_cpu_environment()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing exports are never overwritten")
    catalog = json.loads(args.catalog.read_text())
    if {m["id"] for m in catalog["models"]} != {"N06", "R17", "R8-C25"} or len(catalog["models"]) != 3:
        raise ValueError("The user-authorized catalog must contain exactly three retained models")
    sys.path.insert(0, str(args.source_root / "runtime" / "bundle"))
    from retrain.persistence import load_file, canonical_digest
    from retrain.ema import select_variant, tensor_digest
    from retrain.age import validate_checkpoint
    import torch

    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU initialized before CPU extraction")
    exports = []
    for model in catalog["models"]:
        checkpoint = Path(model["path"])
        sidecar = checkpoint.with_name(checkpoint.name + ".json")
        record = json.loads(sidecar.read_text())
        if model["variant"] != "ema" or record.get("sha256") != model["sha256"]:
            raise ValueError("Catalog and original checkpoint sidecar differ: " + model["id"])
        # Native load_file verifies the complete checkpoint bytes against its sidecar.
        payload = load_file(checkpoint, map_location="cpu")
        if payload.get("successful_updates") != model["updates"] or payload.get("mode") != "train":
            raise ValueError("Retained checkpoint training/update identity differs: " + model["id"])
        job = payload["job"]
        validate_checkpoint(payload, job)
        state = select_variant(payload, "ema")
        tensor_sha = tensor_digest(state)
        if "normalization" not in job or not payload.get("identity"):
            raise ValueError("Missing loading/source identity: " + model["id"])
        parameters = {
            "schema_version": 1,
            "model_id": model["id"],
            "scope": "retained_exact_ema_loading_parameters",
            "usage": "research_only_no_competition_training_source" if model["id"] == "N06" else "retained_single_model",
            "catalog": "records/storage/KEEP-MODELS.json",
            "checkpoint": {"path": str(checkpoint), "sha256": model["sha256"], "variant": "ema",
                           "successful_updates": model["updates"], "tensor_sha256": tensor_sha,
                           "sidecar": record, "sidecar_sha256": sha256(sidecar)},
            "job": job,
            "age": job.get("age"),
            "normalization": job["normalization"],
            "ema": {k: v for k, v in payload["ema"].items() if k != "model"},
            "identity": payload["identity"],
            "manifest_digest": payload["manifest_digest"],
            "schedule": payload.get("schedule"),
            "loading": {"source_kind": "exact_checkpoint_file", "map_location": "cpu", "variant": "ema",
                        "native_loader": "retrain.persistence.load_file",
                        "variant_selector": "retrain.ema.select_variant",
                        "job_sha256": canonical_digest(job), "initialize_public_weights": False,
                        "requires_adjacent_original_sidecar": True},
        }
        # Reject non-JSON state; no tensors, optimizer, RNG or sample predictions are exported.
        json.dumps(parameters, allow_nan=False)
        exports.append(parameters)
        del payload, state, job
        gc.collect()
        print(json.dumps({"model_id": model["id"], "checkpoint_sha256_verified": model["sha256"],
                          "ema_tensor_sha256": tensor_sha, "gpu_initialized": torch.cuda.is_initialized()}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU extraction initialized GPU")
    for parameters in exports:
        write_new(args.output / parameters["model_id"] / "parameters.json", parameters)
    receipt = {
        "schema_version": 1, "status": "completed", "exit_code": 0,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"pod": socket.gethostname(), "uid": os.environ["NCCCU_POD_UID"],
                        "namespace": os.environ["NCCCU_POD_NAMESPACE"], "python": sys.executable,
                        "python_version": sys.version.split()[0],
                        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}},
        "catalog_sha256": sha256(args.catalog), "exporter_sha256": sha256(__file__),
        "loader_source": {str(path.relative_to(args.source_root)): sha256(path) for path in
                          (args.source_root / "runtime/bundle/retrain/persistence.py",
                           args.source_root / "runtime/bundle/retrain/ema.py",
                           args.source_root / "runtime/bundle/retrain/age.py")},
        "files": {str((args.output / p["model_id"] / "parameters.json").relative_to(args.output)):
                  sha256(args.output / p["model_id"] / "parameters.json") for p in exports},
        "checks": ["exact_checkpoint_bytes_match_catalog_and_original_sidecar", "native_cpu_deserialization",
                   "training_mode_and_successful_update_identity", "native_age_contract_validation",
                   "explicit_ema_variant_counters_shapes_dtypes_and_finiteness", "ema_tensor_digest_recorded"],
        "model_forward_performed": False, "official_evaluation_performed": False,
        "weights_copied": False, "gpu_initialized": False, "training_locks_acquired": False,
        "training_signals_sent": False,
    }
    write_new(args.output / "PARAMETERS-RECEIPT.json", receipt)
    print(json.dumps({"status": "completed", "models": len(exports), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
