"""Step 1b — are pinned-pickle predictions independent of the scikit-learn version?

The pickles were fitted under scikit-learn 1.7.0; the twin's own venv has 1.7.2
and the controller venv that produced the manuscript's results has 1.8.0. Run
this script once per interpreter; each run writes
pinned_predictions_sklearn<version>.npz, and every run compares all npz files
present and writes sklearn_version_check.json.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import sklearn

import common as C

warnings.filterwarnings("ignore")


def main():
    df, _ = C.load_runs(verify=False)
    out = {}
    grid = np.linspace(0.0, 6.0, 6001) * 1e6                      # kbit/s, 1 Mbps steps
    for upf, (variant, _) in C.UPFS.items():
        models = [joblib.load(C.PINNED_MODELS / "layer1" / f"{variant}__{t}__lite.pkl") for t in C.L1_TARGETS]
        power = joblib.load(C.PINNED_MODELS / "layer2" / f"{variant}__{C.T_POWER}__lite.pkl")
        for tag, dl in (("samples", df.loc[df.upf == upf, C.DL_TX].to_numpy()), ("grid", grid)):
            X = np.column_stack([dl, dl])                            # twin: UL := DL
            l1 = [m.predict(X) for m in models]
            out[f"{upf}_{tag}"] = np.column_stack([*l1, power.predict(np.column_stack([X, *l1]))])
    np.savez_compressed(C.OUT / f"pinned_predictions_sklearn{sklearn.__version__}.npz", **out)

    files = sorted(C.OUT.glob("pinned_predictions_sklearn*.npz"))
    ref = np.load(files[0])
    report = {"files": [f.name for f in files], "reference": files[0].name, "max_abs_diff": {}}
    for f in files[1:]:
        other = np.load(f)
        report["max_abs_diff"][f.name] = {k: float(np.max(np.abs(ref[k] - other[k]))) for k in ref.files}
    (C.OUT / "sklearn_version_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
