"""
Evaluate and optionally calibrate the phase-correlation prefilter.

Usage:
  # Evaluate PC heuristic against labeled dataset
  python scripts/pc_check.py eval --dataset data/raw/dataset_manual --labels data/raw/dataset_manual/labels.csv --out data/processed/pc_eval.json

  # Fit a calibrator model (logistic regression) and save
  python scripts/pc_check.py calibrate --dataset data/raw/dataset_manual --labels data/raw/dataset_manual/labels.csv --output data/models/pc_calibrator.pkl

"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

# Ensure project root is on sys.path so 'src' package imports work when the
# script is executed as `python scripts/pc_check.py` (sys.path[0] will point to
# scripts/ rather than the repository root).
import sys
from pathlib import Path as _P
proj_root = str(_P(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.features.phase_correlation import check_georeferencing_phase


def eval_pc(dataset_dir: Path, labels_csv: Path, out_json: Path, calibrator_path: Path = None):
    df = pd.read_csv(labels_csv)
    results = []
    y_true = []
    y_score = []
    X = []
    for _, row in df.iterrows():
        oid = str(row['ortho_id']).strip()
        label = int(row['label'])
        ortho_streets = dataset_dir / f"{oid}_ortho_streets.png"
        streets_only = dataset_dir / f"{oid}_streets_only.png"
        ortho_sat = dataset_dir / f"{oid}_ortho_satellite.png"
        sat_only = dataset_dir / f"{oid}_satellite_only.png"

        res = check_georeferencing_phase(
            str(ortho_streets) if ortho_streets.exists() else None,
            str(streets_only) if streets_only.exists() else None,
            str(ortho_sat) if ortho_sat.exists() else None,
            str(sat_only) if sat_only.exists() else None,
            basemap='both',
            use_edges=True,
        )
        prob = float(res.get('combined_good_probability', 0.0))
        # derive aggregate psr and disp if available
        psr_vals = []
        disp_vals = []
        for key in ('streets', 'satellite'):
            if key in res:
                r = res[key]
                psr_vals.append(float(r.get('peak_to_sidelobe_ratio', 0.0)))
                disp_vals.append(float(r.get('shift_magnitude_px', 0.0)))
        psr = float(np.mean(psr_vals)) if psr_vals else 0.0
        disp = float(np.mean(disp_vals)) if disp_vals else 9999.0

        results.append({
            'ortho_id': oid,
            'label': label,
            'pc_prob': prob,
            'psr': psr,
            'disp': disp,
        })
        y_true.append(label)
        y_score.append(prob)
        X.append([psr, disp])

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    X = np.array(X)

    auc = None
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = None

    # simple threshold at 0.5
    preds = (y_score >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, preds)) if len(y_true) > 0 else None

    # if a calibrator is provided, apply it to (psr, disp)
    if calibrator_path is not None and calibrator_path.exists():
        clf = joblib.load(calibrator_path)
        probs_cal = clf.predict_proba(np.array([r for r in X]))[:, 1]
        # replace pc_prob and scores in results
        for i, p in enumerate(probs_cal):
            results[i]['pc_prob_calibrated'] = float(p)
        y_score = probs_cal
        # recompute AUC and accuracy
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None
        preds = (y_score >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, preds)) if len(y_true) > 0 else None

    summary = {'n': len(results), 'auc': auc, 'accuracy_0.5': acc}
    payload = {'summary': summary, 'results': results}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote evaluation to {out_json} - AUC={auc} ACC@0.5={acc}")
    return X, y_true, y_score


def calibrate(dataset_dir: Path, labels_csv: Path, output_path: Path):
    X, y_true, _ = eval_pc(dataset_dir, labels_csv, out_json=Path('/tmp/pc_eval_temp.json'))
    if X.shape[0] == 0:
        print('No samples to calibrate')
        return
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y_true)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Saved calibrator to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--dataset', type=Path, default=Path('data/raw/dataset_manual'))
    p_eval.add_argument('--labels', type=Path, default=Path('data/raw/dataset_manual/labels.csv'))
    p_eval.add_argument('--out', type=Path, default=Path('data/processed/pc_eval.json'))
    p_eval.add_argument('--calibrator', type=Path, default=Path('data/models/pc_calibrator.pkl'))

    p_cal = sub.add_parser('calibrate')
    p_cal.add_argument('--dataset', type=Path, default=Path('data/raw/dataset_manual'))
    p_cal.add_argument('--labels', type=Path, default=Path('data/raw/dataset_manual/labels.csv'))
    p_cal.add_argument('--output', type=Path, default=Path('data/models/pc_calibrator.pkl'))

    args = parser.parse_args()
    if args.cmd == 'eval':
        eval_pc(args.dataset, args.labels, args.out, calibrator_path=args.calibrator)
    elif args.cmd == 'calibrate':
        calibrate(args.dataset, args.labels, args.output)
    else:
        parser.print_help()
