"""
参数敏感分析实验
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PARAM_RANGES = {
    "pop": [(5, 40), (8, 25), (10, 20), (20, 10), (25, 8), (40, 5)],
    "t_mig": [1, 3, 5, 10],
    "s_elite": [1, 3, 5, 10],
}
DEFAULTS = {"pop": (10, 20), "t_mig": 5, "s_elite": 3}

BASE_PARAMS = dict(
    N=10,
    M=20,
    maxGen=50,
    pc=0.9,
    t_mig=5,
    s_elite=3,
    ptrans=0.5,
    mrmr_K=200,
    seed=42,
    random_state=42,
    max_depth=30,
)

HV_REF = (1.0, 1.0)


def value_str(param, value):
    if param == "pop":
        return f"{value[0]}×{value[1]}"  # 用 × 而非 -, 避免 Excel 将 8-25 误识别为日期
    return str(value)


def worker_main(args):
    from sklearn.model_selection import train_test_split
    from metrics import feature_ratio, hypervolume_2d, mce_metric
    from Model import get_model
    from Model.functionHO import build_classifier

    ds_name, param = args.ds, args.param
    value = (
        (tuple(int(x) for x in args.value.split("×")))
        if param == "pop"
        else int(args.value)
    )
    repeats = int(args.repeats)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "Dataset" / ds_name)
    feat = df.iloc[:, :-1].to_numpy(dtype=float)
    label = df.iloc[:, -1].to_numpy().ravel()
    D = feat.shape[1]
    try:
        xtr, xte, ytr, yte = train_test_split(
            feat, label, test_size=0.3, random_state=42, stratify=label
        )
    except ValueError:
        xtr, xte, ytr, yte = train_test_split(
            feat, label, test_size=0.3, random_state=42
        )

    cls_opts = dict(BASE_PARAMS)

    def evaluate(sel_idx):
        if sel_idx.size == 0:
            return 0.0, 1.0
        mdl = build_classifier(cls_opts)
        mdl.fit(xtr[:, sel_idx], np.asarray(ytr).ravel())
        y_pred = mdl.predict(xte[:, sel_idx])
        acc = float(np.mean(np.asarray(yte).ravel() == y_pred))
        return acc, mce_metric(yte, y_pred)

    opts = dict(BASE_PARAMS)
    if param == "pop":
        opts["N"], opts["M"] = value
    else:
        opts[param] = int(value)

    jfs = get_model("rnmpga")
    rep_records, all_obj, curves = [], [], []
    for rep in range(repeats):
        o = dict(opts)
        o["seed"] = BASE_PARAMS["seed"] + rep
        o["fold"] = {"xt": xtr, "yt": ytr, "xv": xte, "yv": yte}
        t0 = time.perf_counter()
        res = jfs(feat, label, o)
        curves.append([float(v) for v in res["c"]] if res.get("c") is not None else [])
        uniq = {}
        for sel in res["pf"]:
            key = tuple(int(i) for i in sel)
            if key not in uniq:
                uniq[key] = evaluate(np.asarray(key, dtype=int))
        obj_uniq = np.array([[len(k) / D, 1.0 - uniq[k][0]] for k in uniq])
        best_key = min(uniq, key=lambda k: uniq[k][1])
        elapsed = time.perf_counter() - t0
        rep_records.append(
            {
                "rep": rep,
                "mce": float(uniq[best_key][1]),
                "fn": float(feature_ratio(len(best_key), D)),
                "nf": int(len(best_key)),
                "hv": float(hypervolume_2d(obj_uniq, HV_REF)),
                "front": obj_uniq.tolist(),
                "runtime": float(elapsed),
            }
        )
        all_obj.append(obj_uniq)

    payload = {
        "dataset": ds_name,
        "param": param,
        "value": value,
        "value_str": value_str(param, value),
        "repeats": repeats,
        "D": int(D),
        "reps": rep_records,
        "true_pf": np.vstack(all_obj).tolist(),
    }
    vs = value_str(param, value).replace("/", "_")
    tmp = outdir / f"combo_{ds_name}_{param}_{vs}.json.tmp"
    final = outdir / f"combo_{ds_name}_{param}_{vs}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)

    c_payload = {
        "dataset": ds_name,
        "param": param,
        "value": value,
        "value_str": value_str(param, value),
        "repeats": repeats,
        "maxgen": len(curves[0]) if curves and curves[0] else None,
        "curves": curves,
        "note": "curves[i] 为第 i 次重复 (repeat) 的逐代最优适应度 "
        "(rnmpga 为分类准确率, 越大越好; 长度 = maxGen)",
    }
    c_tmp = outdir / f"curves_{ds_name}_{param}_{vs}.json.tmp"
    c_final = outdir / f"curves_{ds_name}_{param}_{vs}.json"
    c_tmp.write_text(json.dumps(c_payload, ensure_ascii=False), encoding="utf-8")
    os.replace(c_tmp, c_final)

    print(f"[OK] {ds_name} | {param}={vs}", flush=True)


def run_all(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.datasets is None or args.datasets.strip().lower() == "auto":
        datasets = sorted([p.name for p in (ROOT / "Dataset").glob("*.csv")])
        print(f"自动扫描到 {len(datasets)} 个数据集: {datasets}")
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    params = [p.strip() for p in args.params.split(",") if p.strip()]
    if params == ["auto"]:
        params = list(PARAM_RANGES)
    todo = []
    for ds in datasets:
        for p in params:
            for v in PARAM_RANGES[p]:
                vs = value_str(p, v).replace("/", "_")
                final = outdir / f"combo_{ds}_{p}_{vs}.json"
                if args.resume and final.exists():
                    continue
                todo.append((ds, p, v))
    print(f"待运行 {len(todo)} 组合 (串行)", flush=True)

    failed = 0
    for i, (ds, p, v) in enumerate(todo, 1):
        ok = False
        for attempt in range(1, args.tries + 1):
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--ds",
                        ds,
                        "--param",
                        p,
                        "--value",
                        value_str(p, v),
                        "--repeats",
                        str(args.repeats),
                        "--outdir",
                        str(outdir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[超时] {ds} | {p}={value_str(p, v)} 尝试 {attempt}/{args.tries}",
                    flush=True,
                )
                continue
            if proc.returncode == 0:
                ok = True
                print(f"[{i}/{len(todo)}] {ds} | {p}={value_str(p, v)}", flush=True)
                break
            print(
                f"[失败] {ds} | {p}={value_str(p, v)} 尝试 {attempt}/{args.tries} "
                f"exit={proc.returncode}",
                flush=True,
            )
        if not ok:
            failed += 1
    print(f"完成, 失败 {failed} 个; 结果: {outdir}", flush=True)


def analyze(args):
    outdir = Path(args.outdir)
    valid_vs = {p: {value_str(p, v) for v in PARAM_RANGES[p]} for p in PARAM_RANGES}
    combos = {}
    for p in sorted(outdir.glob("combo_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["param"] not in PARAM_RANGES or d["value_str"] not in valid_vs[d["param"]]:
            continue  # 旧格式结果 (如 pop_8-25 / s_elite_1.0), 跳过
        combos[(d["dataset"], d["param"], d["value_str"])] = d
    if not combos:
        raise SystemExit(f"{outdir} 下没有 combo_*.json")
    datasets = sorted({d for d, _, _ in combos})
    params = [p for p in PARAM_RANGES if any(p == pp for _, pp, _ in combos)]
    print(f"加载 {len(combos)} 组合: {len(datasets)} 数据集 × {len(params)} 参数")

    from metrics import non_dominated_indices, igd

    all_obj = [np.asarray(r["front"]) for c in combos.values() for r in c["reps"]]
    shared = np.vstack(all_obj)
    shared_pf = shared[non_dominated_indices(shared)]
    for (ds, p, vs), c in combos.items():
        c["true_pf"] = np.asarray(c["true_pf"])[
            non_dominated_indices(np.asarray(c["true_pf"]))
        ]
        for r in c["reps"]:
            r["igd_own"] = float(igd(np.asarray(r["front"]), c["true_pf"]))
            r["igd_shared"] = float(igd(np.asarray(r["front"]), shared_pf))

    rows = []
    for ds in datasets:
        for p in params:
            for v in PARAM_RANGES[p]:
                vs = value_str(p, v)
                key = (ds, p, vs)
                if key not in combos:
                    continue
                c = combos[key]
                row = {
                    "Dataset": ds,
                    "Param": p,
                    "Value": vs,
                    "IsDefault": (vs == value_str(p, DEFAULTS[p])),
                }
                for m_ in ("mce", "fn", "hv", "igd_own", "igd_shared", "nf", "runtime"):
                    arr = np.asarray([r[m_] for r in c["reps"]], dtype=float)
                    row[f"{m_}_mean"] = float(arr.mean())
                    row[f"{m_}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                rows.append(row)
    summary = pd.DataFrame(rows).round(6)
    summary.to_excel(outdir / "summary.xlsx", index=False)
    print(f"结果已导出: {outdir / 'summary.xlsx'}")


def main():
    parser = argparse.ArgumentParser(description="RNMPGA 参数敏感分析")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--ds", default="")
    parser.add_argument("--param", default="")
    parser.add_argument("--value", default="")
    parser.add_argument("--repeats", default=3)
    parser.add_argument(
        "--datasets", default="auto", help="数据集列表，用逗号分隔，或 'auto' 自动扫描"
    )
    parser.add_argument("--params", default="auto")
    parser.add_argument("--outdir", default="Results/exp_sensitivity")
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args.outdir = (
        str(ROOT / args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    )
    if args.analyze:
        analyze(args)
    elif args.worker:
        worker_main(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
