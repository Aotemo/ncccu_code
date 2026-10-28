"""Narrow Pod-only checks for repository navigation, retained parameters and CPU transport."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


DOCS = (
    "README.md", "AGENTS.md", "docs/LAYOUT.md", "docs/SOURCE-MAP.md", "docs/EXECUTION.md", "docs/CODE-SYNC.md", "docs/STORAGE.md",
    "todo/README.md", "todo/auroc-999/README.md", "todo/storage-retention/README.md", "models/README.md",
    "configs/README.md", "records/README.md", "runtime/README.md", "tools/README.md",
    "todo/repository-layout/README.md", "records/repository-layout/STATUS.md",
    "models/N06/README.md", "models/R17/README.md", "models/R8-C25/README.md",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--require-checkpoint-export", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if socket.gethostname() != os.environ.get("NCCCU_POD_NAME") or not os.environ.get("NCCCU_POD_UID"):
        raise RuntimeError("Checks must run on a freshly verified compute Pod")
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("Checks require CPU priority 19")
    if any(os.environ.get(key) != "" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise RuntimeError("Checks require hidden accelerators")
    checks = []
    for relative in DOCS:
        path = root / relative
        content = path.read_text()
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0].strip("<>")
            if not target or re.match(r"[a-z]+://", target):
                continue
            if not (path.parent / target).exists():
                raise ValueError(f"Broken documentation link: {relative} -> {target}")
        checks.append({"name": "documentation_links:" + relative, "passed": True})
    for relative in ("README.md", "AGENTS.md", "docs/EXECUTION.md", "tools/environment/pod_cpu.sh",
                     "tools/models/export_parameters.py"):
        content = (root / relative).read_text()
        if re.search(r"vcjob-\d|/usr/local/python\d|/public/home/|/Users/|srv_[A-Za-z0-9]+|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", content):
            raise ValueError("Fixed machine identity in reusable entry: " + relative)
    checks.append({"name": "reusable_entries_have_no_fixed_machine_identity", "passed": True})
    subprocess.run(["bash", "-n", str(root / "tools/environment/pod_cpu.sh")], check=True)
    checks.append({"name": "cpu_transport_shell_syntax", "passed": True})
    for relative in ("tools/models/export_parameters.py", "tools/repository/check_structure.py"):
        compile((root / relative).read_text(), relative, "exec")
    checks.append({"name": "changed_python_syntax_without_execution_or_bytecode", "passed": True})
    catalog = json.loads((root / "records/storage/KEEP-MODELS.json").read_text())
    ids = {model["id"] for model in catalog["models"]}
    if ids != {"N06", "R17", "R8-C25"} or len(catalog["models"]) != 3:
        raise ValueError("Retention catalog must contain exactly the three authorized models")
    metadata_files = []
    for model in catalog["models"]:
        directory = root / "models" / model["id"]
        note = json.loads((directory / "MODEL.json").read_text())
        if any(note.get(key) != model[key] for key in ("id", "path", "sha256", "variant", "updates")):
            raise ValueError("Retained model note differs from catalog: " + model["id"])
        records = {}
        for kind in ("job", "config", "contract", "source-files"):
            paths = list(directory.glob("*-" + kind + ".json"))
            if len(paths) != 1:
                raise ValueError("Expected one original " + kind + " for " + model["id"])
            records[kind] = json.loads(paths[0].read_text())
            metadata_files.append(str(paths[0].relative_to(root)))
        job = records["job"]
        identity = records["contract"]["jobs"][job["name"]]
        if any(job[key] != identity["job"][key] for key in ("model", "normalization", "age")):
            raise ValueError("Original job and identity differ: " + model["id"])
        if not isinstance(records["source-files"], dict) or not records["source-files"]:
            raise ValueError("Original source identity missing: " + model["id"])
        if model["id"] == "R8-C25" and job["age"]["enabled"] is not False:
            raise ValueError("R8-C25 must remain image-only")
        metadata_files.append(str((directory / "MODEL.json").relative_to(root)))
        checks.append({"name": "retained_original_metadata_consistency:" + model["id"], "passed": True})
    pending = []
    export_receipt_path = root / "models/PARAMETERS-RECEIPT.json"
    if export_receipt_path.exists():
        receipt = json.loads(export_receipt_path.read_text())
        if receipt.get("status") != "completed" or receipt.get("gpu_initialized") is not False:
            raise ValueError("Missing real successful CPU parameter receipt")
        if receipt["catalog_sha256"] != sha256(root / "records/storage/KEEP-MODELS.json"):
            raise ValueError("Parameter receipt does not bind the actual catalog")
        if receipt["exporter_sha256"] != sha256(root / "tools/models/export_parameters.py"):
            raise ValueError("Parameter receipt does not bind the actual exporter")
        if set(receipt["files"]) != {model_id + "/parameters.json" for model_id in ids}:
            raise ValueError("Parameter receipt model file list differs")
        for relative, expected in receipt["loader_source"].items():
            if sha256(root / relative) != expected:
                raise ValueError("Loading source changed since extraction: " + relative)
        for model in catalog["models"]:
            relative = model["id"] + "/parameters.json"
            parameters = json.loads((root / "models" / relative).read_text())
            checkpoint = parameters["checkpoint"]
            if (parameters["model_id"] != model["id"] or checkpoint["path"] != model["path"]
                    or checkpoint["sha256"] != model["sha256"] or checkpoint["variant"] != model["variant"]
                    or checkpoint["successful_updates"] != model["updates"]):
                raise ValueError("Exported loading identity differs: " + model["id"])
            if parameters["normalization"] != parameters["job"]["normalization"] or parameters["age"] != parameters["job"].get("age"):
                raise ValueError("Exported parameters are inconsistent: " + model["id"])
            if model["id"] == "N06" and parameters["usage"] != "research_only_no_competition_training_source":
                raise ValueError("N06 research isolation absent")
            if sha256(root / "models" / relative) != receipt["files"][relative]:
                raise ValueError("Parameter file differs from extraction receipt: " + model["id"])
            metadata_files.append("models/" + relative)
        metadata_files.append("models/PARAMETERS-RECEIPT.json")
        checks.append({"name": "actual_checkpoint_export_receipt_consistency", "passed": True})
    elif args.require_checkpoint_export:
        raise ValueError("Checkpoint extraction has not been performed")
    else:
        pending.append("checkpoint_parameter_extraction_and_cpu_loading_validation_not_performed")
    files = [*DOCS, "tools/environment/pod_cpu.sh", "tools/models/export_parameters.py", "tools/repository/check_structure.py",
             *metadata_files]
    result = {"status": "completed_with_pending_checkpoint_validation" if pending else "completed", "exit_code": 0,
              "scope": "repository_navigation_dynamic_entry_and_retained_parameter_receipt",
              "environment": {"pod": socket.gethostname(), "uid": os.environ["NCCCU_POD_UID"], "python": sys.executable},
              "checks": checks, "pending_checks": pending, "files": {relative: sha256(root / relative) for relative in files},
              "model_forward_performed": False, "official_evaluation_performed": False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks), "pending_checks": pending, "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
