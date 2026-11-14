#!/usr/bin/env python3
"""
convert_to_dlc.py - convert ONNX models to DLC with whichever SNPE SDK is active.

1. Detects the SNPE version of the current container (1.61 or 2.22).
2. Converts each ONNX model with that SDK's snpe-onnx-to-dlc.
3. Saves the DLC in the matching folder:  ./1.6  (SNPE 1.x)  or  ./2.22  (SNPE 2.22),
   named <model>_<YYYYMMDD_HHMMSS>_conversion_<1.6|2.22>.dlc, together with a
   conversion log and the snpe-dlc-info output.
   A model that already has a DLC in that folder is reported as "already converted".

Works in both containers (Python 3.6 in the 1.61 container, 3.10 in the 2.22 one).

Usage (run inside a container, from the repo folder):
    python3 convert_to_dlc.py                     # convert every *.onnx in this folder
    python3 convert_to_dlc.py jacob_best_model.onnx rogers_model.onnx
    python3 convert_to_dlc.py --force             # convert again even if already converted
    python3 convert_to_dlc.py --no-preserve-io    # 2.22 only: let the converter change input layout
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ONNX limits for each converter (from the SDK setup notes)
LIMITS = {1: {"opset": 11, "ir": None}, 2: {"opset": 17, "ir": 8}}


def run(cmd, log_file=None):
    """Run a command (Python 3.6 compatible) and return (exit code, output text)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if log_file:
        Path(log_file).write_text(" ".join(cmd) + "\n\n" + p.stdout)
    return p.returncode, p.stdout


def detect_snpe():
    """Return (major, minor, full_version) of the active SNPE SDK."""
    root = os.environ.get("SNPE_ROOT", "")
    m = re.search(r"(\d+)\.(\d+)\.(\d+)(?:\.\d+)?", os.path.basename(root.rstrip("/")))
    if not m and shutil.which("snpe-net-run"):
        _, out = run(["snpe-net-run", "--version"])
        m = re.search(r"SNPE v?(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        sys.exit("ERROR: could not detect the SNPE version. Are you inside a SNPE container "
                 "(echo $SNPE_ROOT should show the SDK path)?")
    return int(m.group(1)), int(m.group(2)), m.group(0)


def onnx_info(path):
    """Return (ir_version, opset, [(input_name, [dims])]) or None if the onnx package is missing."""
    try:
        import onnx
    except ImportError:
        return None
    m = onnx.load(str(path))
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    init_names = {i.name for i in m.graph.initializer}
    inputs = []
    for i in m.graph.input:
        if i.name in init_names:
            continue
        dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or "?") for d in i.type.tensor_type.shape.dim]
        inputs.append((i.name, dims))
    return m.ir_version, opset, inputs


def convert(onnx_path, major, out_dir, force, preserve_io):
    name = onnx_path.stem
    tag = out_dir.name                      # "1.6" or "2.22"
    existing = sorted(out_dir.glob("{}_*_conversion_{}.dlc".format(name, tag)))
    if existing and not force:
        return "already converted", str(existing[-1])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "{}_{}_conversion_{}".format(name, stamp, tag)
    dlc = out_dir / (base + ".dlc")
    log = out_dir / (base + "_convert.log")
    info_txt = out_dir / (base + "_info.txt")

    cmd = ["snpe-onnx-to-dlc", "--input_network", str(onnx_path), "--output_path", str(dlc)]

    info = onnx_info(onnx_path)
    if info:
        ir, opset, inputs = info
        lim = LIMITS[major]
        if opset > lim["opset"] or (lim["ir"] and ir > lim["ir"]):
            return "not converted", "IR {} / opset {} is above this SDK's limit (opset <= {}{})".format(
                ir, opset, lim["opset"], ", IR <= {}".format(lim["ir"]) if lim["ir"] else "")
        for in_name, dims in inputs:
            if any(not isinstance(d, int) for d in dims):
                fixed = [d if isinstance(d, int) else 1 for d in dims]
                cmd += ["-d", in_name, ",".join(str(d) for d in fixed)]
                print("   note: input '{}' has dynamic dims {}, converting with {}".format(in_name, dims, fixed))
        if major == 2 and preserve_io and inputs:
            cmd += ["--preserve_io", "layout"] + [n for n, _ in inputs]
    else:
        print("   note: 'onnx' package not available, skipping version and input checks")

    code, out = run(cmd, log)
    if code != 0 or not dlc.exists():
        last = [l for l in out.strip().splitlines() if l.strip()][-3:]
        return "failed", "see {}: {}".format(log.name, " | ".join(last))

    run(["snpe-dlc-info", "-i", str(dlc)], info_txt)
    warnings = sum(1 for l in out.splitlines() if "WARNING" in l)
    msg = "{} ({:,} bytes)".format(dlc, dlc.stat().st_size)
    if warnings:
        msg += ", {} warning(s) in {}".format(warnings, log.name)
    return "converted", msg


def main():
    ap = argparse.ArgumentParser(description="Convert ONNX models to DLC with the active SNPE SDK.")
    ap.add_argument("models", nargs="*", help="ONNX files (default: every *.onnx in --dir)")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder with the ONNX files and the 1.6 / 2.22 output folders (default: script folder)")
    ap.add_argument("--force", action="store_true", help="convert again even if a DLC already exists (new timestamped file)")
    ap.add_argument("--no-preserve-io", action="store_true",
                    help="SNPE 2.x only: do not keep the original input layout")
    args = ap.parse_args()

    if not shutil.which("snpe-onnx-to-dlc"):
        sys.exit("ERROR: snpe-onnx-to-dlc not found. Source the SDK first, e.g. "
                 "source $SNPE_ROOT/bin/envsetup.sh")

    major, minor, full = detect_snpe()
    if major not in LIMITS:
        sys.exit("ERROR: unsupported SNPE version {}".format(full))
    base = Path(args.dir).resolve()
    out_dir = base / ("1.6" if major == 1 else "{}.{}".format(major, minor))
    out_dir.mkdir(exist_ok=True)

    def find(m):
        p = Path(m)
        return (p if p.exists() or p.is_absolute() else base / p).resolve()
    models = [find(m) for m in args.models] if args.models else sorted(base.glob("*.onnx"))
    if not models:
        sys.exit("No ONNX files found in {}".format(base))

    print("SNPE version : {}".format(full))
    print("Output folder: {}".format(out_dir))
    print("Models       : {}\n".format(", ".join(m.name for m in models)))

    results = []
    for m in models:
        print("-> {}".format(m.name))
        if not m.exists():
            results.append((m.name, "failed", "file not found"))
            continue
        status, msg = convert(m, major, out_dir, args.force, not args.no_preserve_io)
        if status == "already converted":
            print("   Already converted: {}".format(msg))
        elif status == "converted":
            print("   Converted. Stored at: {}".format(msg))
        else:
            print("   {}: {}".format(status, msg))
        results.append((m.name, status, msg))

    print("\nSummary (SNPE {})".format(full))
    w = max(len(r[0]) for r in results)
    for name, status, msg in results:
        print("  {:<{w}}  {:<18} {}".format(name, status, msg, w=w))
    sys.exit(1 if any(r[1] == "failed" for r in results) else 0)


if __name__ == "__main__":
    main()
