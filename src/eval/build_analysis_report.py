"""
Assemble the interactive analysis report (single HTML file) from the cache
written by ``build_analysis_data.py``.

    python src/eval/build_analysis_report.py \
        --cache eval_results/analysis_cache --out eval_results/analysis_report.html

Layout of the embedded data (all binary blobs are gzip + base64, decoded in the
browser with ``DecompressionStream``):

* ``grp``    float32 [system, epoch, testset, group, metric, seed]  (NaN = missing)
* ``ia/it/is/isn``  uint8 [system, item_epoch, item]  seed-counts of correct
  (acc / acc_tot / acc_span) and of defined span comparisons
* ``im``     int8  [system, item_epoch, item]  mean per-token span margin * MARGIN_SCALE
* ``items``  json  sentence texts and span strings
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os

import numpy as np

MARGIN_SCALE = 12.0
HERE = os.path.dirname(os.path.abspath(__file__))


def pack(arr: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="eval_results/analysis_cache")
    ap.add_argument("--out", default="eval_results/analysis_report.html")
    ap.add_argument("--template", default=os.path.join(HERE, "report_template.html"))
    ap.add_argument("--findings", default=os.path.join(HERE, "report_findings.html"))
    ap.add_argument("--css", default=os.path.join(HERE, "report_style.css"))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, "meta.json"), encoding="utf-8"))
    b = np.load(os.path.join(args.cache, "bundle.npz"))
    items = meta.pop("items")

    blobs = {
        "grp": pack(b["grp"].astype("<f4")),
        "ia": pack(b["item_acc"]),
        "it": pack(b["item_tot"]),
        "is": pack(b["item_span"]),
        "isn": pack(b["item_span_n"]),
        "im": pack(np.clip(np.round(b["item_margin"] * MARGIN_SCALE), -127, 127).astype(np.int8)),
        "items": base64.b64encode(
            gzip.compress(json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode(), 9)
        ).decode(),
    }
    meta["margin_scale"] = MARGIN_SCALE
    meta["grp_shape"] = list(b["grp"].shape)
    meta["n_items"] = len(items["gold"])

    html = open(args.template, encoding="utf-8").read()
    css = open(args.css, encoding="utf-8").read()
    html = html.replace("/*__CSS__*/", css)
    if os.path.isfile(args.findings):
        html = html.replace("<!--__FINDINGS__-->", open(args.findings, encoding="utf-8").read())
    html = html.replace("__META__", json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__BLOBS__", json.dumps(blobs, separators=(",", ":")))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
