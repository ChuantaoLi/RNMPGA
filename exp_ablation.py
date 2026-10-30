"""
消融分析实验
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

LEVELS = [0, 1, 2, 3, 4]

LEVEL_DESC = {
    0: "L0 基础遗传算法",
    1: "L1 +NSGA-II环境选择",
    2: "L2 +mRMR指导",
    3: "L3 +加权和分解与多种群",
    4: "L4 完整模型",
}

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


# ---------------------------------------------------------------------------
# 各级所需的算子实现
# ---------------------------------------------------------------------------
def make_uniform_init():

    def _init(N, D, g, rng):
        lam = max(2, int(round(D / 200.0)))
        p = np.clip(np.full(D, lam / D), 0.0, 1.0)
        pop = rng.random((N, D)) < p
        none = pop.sum(axis=1) == 0
        if none.any():
            pop[none, rng.integers(0, D, size=int(none.sum()))] = True
        return pop

    return _init


def make_uniform_mutate():

    def _mutate(pop, g, rng, mu=1.0):
        p = np.clip(mu / pop.shape[1], 0.0, 1.0)
        return pop ^ (rng.random(pop.shape) < p)

    return _mutate


def make_truncate_by_err():

    def _sel(pop, obj, accs, off, obj_o, acc_o, N):
        obj_all = np.vstack([obj, obj_o])
        pop2 = np.vstack([pop, off])
        acc_all = np.concatenate([accs, acc_o])
        keep = np.argsort(obj_all[:, 1])[:N]
        return pop2[keep], obj_all[keep], acc_all[keep]

    return _sel


def make_tournament_offspring(use_feat=True, exclude_empty=True):
    import Model.rnmpga as db

    def _to(pop, obj, w, N, pc, rng, D):
        err = obj[:, 1]
        feat = obj[:, 0] / D
        if use_feat:
            fit = w * err + (1.0 - w) * feat
        else:
            fit = err.copy()
        if exclude_empty:
            fit[obj[:, 0] <= 0] = np.inf
        n = pop.shape[0]

        def pick():
            i, j = rng.choice(n, 2, replace=False)
            return i if fit[i] < fit[j] else j

        off = []
        while len(off) < N:
            p1 = pop[pick()].copy()
            p2 = pop[pick()].copy()
            if rng.random() < pc:
                c1, c2 = db.uniform_crossover(p1, p2, rng)
                off.append(c1)
                off.append(c2)
            else:
                off.append(p1)
                off.append(p2)
        return np.vstack(off[:N])

    return _to


def make_neighborhood_mating(use_feat=True, exclude_empty=True, nsga2=True):
    import Model.rnmpga as db

    def _nm(pop, obj, accs, immigrants, ptrans, pc, rng, acc_of, N, D, w):
        if not immigrants:
            return pop, obj, accs
        im_masks = np.vstack([np.asarray(m[0], dtype=bool) for m in immigrants])
        im_n = np.array([m[1] for m in immigrants])
        im_acc = np.array([m[2] for m in immigrants])
        if use_feat:
            im_fit = w * (1.0 - im_acc) + (1.0 - w) * (im_n / D)
        else:
            im_fit = (1.0 - im_acc).copy()
        if exclude_empty:
            im_fit[im_n <= 0] = np.inf
        off = []
        for v in pop:
            if rng.random() < ptrans:
                if len(im_masks) == 1:
                    base = im_masks[0]
                else:
                    a, b = rng.choice(len(im_masks), 2, replace=False)
                    base = im_masks[a if im_fit[a] <= im_fit[b] else b]
                c1, _ = db.uniform_crossover(v, base, rng)
                off.append(c1)
        if not off:
            return pop, obj, accs
        off = np.vstack(off)
        obj_o, acc_o = db.evaluate_pop(off, acc_of)
        obj_all = np.vstack([obj, obj_o])
        if nsga2:
            rank, cd = db.rank_and_crowding(obj_all)
            keep = np.lexsort((-cd, rank))[:N]
        else:
            keep = np.argsort(obj_all[:, 1])[:N]
        pop2 = np.vstack([pop, off])
        return pop2[keep], obj_all[keep], np.concatenate([accs, acc_o])[keep]

    return _nm


def apply_level(db, real, level, opts):
    opts["M"] = 1
    opts["t_mig"] = 10**9
    opts["ptrans"] = 0.0
    db.mrmr_guidance = lambda xt, y, K=200: np.zeros(xt.shape[1])
    db.guided_init = make_uniform_init()
    db.guided_mutate = make_uniform_mutate()
    db.tournament_offspring = make_tournament_offspring(
        use_feat=False, exclude_empty=False
    )
    db.neighborhood_mating = make_neighborhood_mating(
        use_feat=False, exclude_empty=False
    )
    db.nsga2_select = make_truncate_by_err()

    if level >= 1:
        db.nsga2_select = real["nsga2_select"]
    if level >= 2:
        db.mrmr_guidance = real["mrmr_guidance"]
        db.guided_init = real["guided_init"]
        db.guided_mutate = real["guided_mutate"]
    if level >= 3:
        opts["M"] = 20
        db.tournament_offspring = make_tournament_offspring(
            use_feat=True, exclude_empty=False
        )
        db.neighborhood_mating = make_neighborhood_mating(
            use_feat=True, exclude_empty=False
        )
    if level >= 4:
        opts["t_mig"] = 5
        opts["ptrans"] = 0.5
        db.tournament_offspring = real["tournament_offspring"]
        db.neighborhood_mating = real["neighborhood_mating"]


def worker_main(args):
    from sklearn.model_selection import train_test_split
    from metrics import feature_ratio, hypervolume_2d, mce_metric
    from Model import get_model
    from Model.functionHO import build_classifier
    import Model.rnmpga as db

    ds_name, level = args.ds, int(args.level)
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

    real = {
        name: getattr(db, name)
        for name in (
            "mrmr_guidance",
            "guided_init",
            "guided_mutate",
            "tournament_offspring",
            "neighborhood_mating",
            "nsga2_select",
        )
    }
    opts = dict(BASE_PARAMS)
    apply_level(db, real, level, opts)
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
        "level": level,
        "level_desc": LEVEL_DESC[level],
        "repeats": repeats,
        "D": int(D),
        "reps": rep_records,
        "true_pf": np.vstack(all_obj).tolist(),
    }
    tmp = outdir / f"combo_{ds_name}_L{level}.json.tmp"
    final = outdir / f"combo_{ds_name}_L{level}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)

    c_payload = {
        "dataset": ds_name,
        "level": level,
        "level_desc": LEVEL_DESC[level],
        "repeats": repeats,
        "maxgen": len(curves[0]) if curves and curves[0] else None,
        "curves": curves,
        "note": "curves[i] 为第 i 次重复 (repeat) 的逐代最优适应度 "
        "(rnmpga 为分类准确率, 越大越好; 长度 = maxGen)",
    }
    c_tmp = outdir / f"curves_{ds_name}_L{level}.json.tmp"
    c_final = outdir / f"curves_{ds_name}_L{level}.json"
    c_tmp.write_text(json.dumps(c_payload, ensure_ascii=False), encoding="utf-8")
    os.replace(c_tmp, c_final)

    print(f"[OK] {ds_name} | L{level}", flush=True)


def run_all(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.datasets is None or args.datasets.strip().lower() == "auto":
        datasets = sorted([p.name for p in (ROOT / "Dataset").glob("*.csv")])
        print(f"自动扫描到 {len(datasets)} 个数据集: {datasets}")
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    todo = []
    for ds in datasets:
        for level in LEVELS:
            final = outdir / f"combo_{ds}_L{level}.json"
            if args.resume and final.exists():
                try:
                    meta = json.loads(final.read_text(encoding="utf-8"))
                    if meta.get("level_desc") == LEVEL_DESC[level]:
                        continue
                except Exception:
                    continue
            todo.append((ds, level))

    print(f"待运行 {len(todo)} 组合 (串行)", flush=True)
    failed = 0
    for i, (ds, level) in enumerate(todo, 1):
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
                        "--level",
                        str(level),
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
                print(f"[超时] {ds} | L{level} 尝试 {attempt}/{args.tries}", flush=True)
                continue
            if proc.returncode == 0:
                ok = True
                print(f"[{i}/{len(todo)}] {ds} | L{level}", flush=True)
                break
            print(
                f"[失败] {ds} | L{level} 尝试 {attempt}/{args.tries} "
                f"exit={proc.returncode}",
                flush=True,
            )
        if not ok:
            failed += 1
    print(f"完成, 失败 {failed} 个; 结果: {outdir}", flush=True)


def analyze(args):
    outdir = Path(args.outdir)
    combos = {}
    for p in sorted(outdir.glob("combo_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["level"] not in LEVELS:
            continue  # 旧协议遗留文件 (如旧 L4/L5), 跳过
        combos[(d["dataset"], d["level"])] = d
    if not combos:
        raise SystemExit(f"{outdir} 下没有 combo_*.json")
    datasets = sorted({d for d, _ in combos})
    levels = sorted({l for _, l in combos})
    print(f"加载 {len(combos)} 组合: {len(datasets)} 数据集 × {len(levels)} 层级")

    from metrics import non_dominated_indices, igd

    all_obj = [np.asarray(r["front"]) for c in combos.values() for r in c["reps"]]
    shared = np.vstack(all_obj)
    shared_pf = shared[non_dominated_indices(shared)]
    for (ds, level), c in combos.items():
        c["true_pf"] = np.asarray(c["true_pf"])[
            non_dominated_indices(np.asarray(c["true_pf"]))
        ]
        for r in c["reps"]:
            r["igd_own"] = float(igd(np.asarray(r["front"]), c["true_pf"]))
            r["igd_shared"] = float(igd(np.asarray(r["front"]), shared_pf))

    rows = []
    for ds in datasets:
        for level in levels:
            c = combos[(ds, level)]
            row = {"Dataset": ds, "Level": f"L{level}", "Desc": c["level_desc"]}
            for m_ in ("mce", "fn", "hv", "igd_own", "igd_shared", "nf", "runtime"):
                arr = np.asarray([r[m_] for r in c["reps"]], dtype=float)
                row[f"{m_}_mean"] = float(arr.mean())
                row[f"{m_}_std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            rows.append(row)
    summary = pd.DataFrame(rows).round(6)

    for k in levels:
        if k == 0:
            continue
        prev = summary[summary.Level == f"L{k-1}"].set_index("Dataset")
        cur = summary[summary.Level == f"L{k}"].set_index("Dataset")
        for m_ in ("mce", "fn", "hv", "igd_own"):
            summary.loc[summary.Level == f"L{k}", f"{m_}_inc"] = (
                cur[f"{m_}_mean"].to_numpy() - prev[f"{m_}_mean"].to_numpy()
            ).round(6)
    summary.to_excel(outdir / "summary.xlsx", index=False)
    print(f"结果已导出: {outdir / 'summary.xlsx'}")


def main():
    parser = argparse.ArgumentParser(description="RNMPGA 递进式消融分析")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--ds", default="")
    parser.add_argument("--level", default="")
    parser.add_argument("--repeats", default=3)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--N", type=int, default=None, help="每子种群规模 (默认 40)")
    parser.add_argument("--maxGen", type=int, default=None, help="最大代数 (默认 40)")
    parser.add_argument("--outdir", default="Results/exp_ablation")
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    if args.N is not None:
        BASE_PARAMS["N"] = args.N
    if args.maxGen is not None:
        BASE_PARAMS["maxGen"] = args.maxGen
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
