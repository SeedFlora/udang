"""Compute paired cluster bootstrap comparisons, seed contrasts, and Holm adjustment."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MANIFEST, METRICS, REPO, attach_groups, load_preds, rel, require, write_table
sys.path.insert(0, str(REPO))
from src.evaluation.bootstrap import paired_grouped_bootstrap, wilcoxon_across_folds
DISEASE_COMPARISONS = [('E2_pooled_bce_ordinary', 'E1_disease_single_tiger', 'pooled vs single-dataset (TigerShrimpBD)'), ('E2_pooled_bce_ordinary', 'E1_disease_single_sdibd', 'pooled vs single-dataset (ShrimpDiseaseImageBD)'), ('E2_pooled_asl_ordinary', 'E2_pooled_bce_ordinary', 'masked asymmetric loss vs BCE'), ('E2_pooled_focal_ordinary', 'E2_pooled_bce_ordinary', 'masked focal loss vs BCE'), ('E2_pooled_bce_balanced', 'E2_pooled_bce_ordinary', 'source-balanced vs ordinary sampling'), ('E2_pooled_bce_withaug', 'E2_pooled_bce_ordinary', 'with published augmented copies vs originals only'), ('E2_pooled_bce_strongcolor', 'E2_pooled_bce_ordinary', 'aggressive colour augmentation vs disease-safe'), ('E6_multitask_fixed', 'E2_pooled_bce_ordinary', 'multi-task vs single-task disease'), ('E6_multitask_uncertainty', 'E6_multitask_fixed', 'uncertainty weighting vs fixed task weights'), ('E7_mtl_mixstyle', 'E6_multitask_fixed', 'MixStyle vs ERM multi-task'), ('E7_mtl_coral', 'E6_multitask_fixed', 'CORAL vs ERM multi-task'), ('E8_lightweight_disease', 'E2_pooled_bce_ordinary', 'MobileNetV3-Large vs ConvNeXt-Tiny')]
BIOMASS_COMPARISONS = [('E5_biomass_multiview_mean', 'E4_biomass_single_image', 'multi-view mean pooling vs single image'), ('E5_biomass_multiview_attention', 'E5_biomass_multiview_mean', 'attention pooling vs mean pooling'), ('E6_multitask_fixed', 'E4_biomass_single_image', 'multi-task biomass vs single-task biomass'), ('E8_lightweight_biomass', 'E4_biomass_single_image', 'MobileNetV3-Large vs ConvNeXt-Tiny (biomass)'), ('E13_biomass_within_db', 'E4_biomass_single_image', 'within-database CV vs leave-one-DB-out (same model; protocol effect)'), ('E12_biomass_loso', 'E4_biomass_single_image', 'leave-one-site-out vs leave-one-DB-out')]
CONFIRMATORY = {('E2_pooled_bce_withaug', 'E2_pooled_bce_ordinary'), ('E6_multitask_fixed', 'E2_pooled_bce_ordinary'), ('E6_multitask_uncertainty', 'E6_multitask_fixed'), ('E7_mtl_mixstyle', 'E6_multitask_fixed'), ('E7_mtl_coral', 'E6_multitask_fixed'), ('E8_lightweight_disease', 'E2_pooled_bce_ordinary'), ('E6_multitask_fixed', 'E4_biomass_single_image'), ('E8_lightweight_biomass', 'E4_biomass_single_image'), ('E13_biomass_within_db', 'E4_biomass_single_image'), ('E12_biomass_loso', 'E4_biomass_single_image')}
TAG_RULE = 'Confirmatory: fixed contrast set; exploratory: remaining contrasts.'
DEFAULT_SEEDS = (42, 2026, 3407)
N_BOOT = 2000
BOOT_SEED = 42
WILCOXON_METRIC = {'disease': 'health_bal_acc', 'biomass': 'ind_weight_mae_g'}
METRIC_NAME = {'disease': 'health_balanced_accuracy_at_0.50', 'biomass': 'individual_weight_mae_g'}
BETTER = {'disease': 'higher', 'biomass': 'lower'}

def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    pred = (np.asarray(p, dtype=float) >= 0.5).astype(int)
    y = np.asarray(y, dtype=float)
    out = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            return float('nan')
        out.append((pred[m] == c).mean())
    return float(np.mean(out))

def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p, dtype=float) - np.asarray(y, dtype=float))))

def paired_bootstrap(y, pa, pb, groups, metric, n_boot: int=N_BOOT, seed: int=BOOT_SEED) -> dict:
    r = paired_grouped_bootstrap(metric, np.asarray(y), np.asarray(pa), np.asarray(pb), np.asarray(groups), n_boot=n_boot, seed=seed)
    p = r['p_two_sided']
    if np.isfinite(p):
        p = float(min(1.0, max(p, 1.0 / n_boot)))
    return {'delta': float(r['diff']), 'lo': float(r['lo']), 'hi': float(r['hi']), 'p_two_sided': p, 'n_groups': int(r['n_groups']), 'metric_a': float(r['metric_a']), 'metric_b': float(r['metric_b']), 'n_boot': int(n_boot)}

def holm(pvals) -> list[float]:
    vals = [float(v) for v in pvals]
    finite = [i for i, v in enumerate(vals) if np.isfinite(v)]
    out = [float('nan')] * len(vals)
    m = len(finite)
    running = 0.0
    for rank, i in enumerate(sorted(finite, key=lambda j: vals[j])):
        running = max(running, (m - rank) * vals[i])
        out[i] = min(1.0, running)
    return out

def pooled_seed_bootstrap(y, preds_a, preds_b, groups, metric, n_boot: int=N_BOOT, seed: int=BOOT_SEED, alpha: float=0.05) -> dict:
    """Use one shared cluster resample per replicate, then average seed deltas."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    pa = [np.asarray(v, dtype=float) for v in preds_a]
    pb = [np.asarray(v, dtype=float) for v in preds_b]
    _, codes = np.unique(groups, return_inverse=True)
    codes = np.asarray(codes).ravel()
    n_groups = int(codes.max()) + 1 if codes.size else 0
    order = np.argsort(codes, kind='stable')
    edges = np.searchsorted(codes[order], np.arange(n_groups + 1))
    group_rows = [order[edges[g]:edges[g + 1]] for g in range(n_groups)]
    point = float(np.mean([metric(y, a) - metric(y, b) for a, b in zip(pa, pb)]))
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        rows = np.concatenate([group_rows[g] for g in rng.integers(0, n_groups, size=n_groups)])
        yt = y[rows]
        reps[i] = float(np.mean([metric(yt, a[rows]) - metric(yt, b[rows]) for a, b in zip(pa, pb)]))
    valid = reps[np.isfinite(reps)]
    if valid.size:
        lo = float(np.percentile(valid, 100.0 * alpha / 2.0))
        hi = float(np.percentile(valid, 100.0 * (1.0 - alpha / 2.0)))
    else:
        lo = hi = float('nan')
    return {'delta': point, 'lo': lo, 'hi': hi, 'n_groups': int(n_groups), 'n_valid': int(valid.size), 'n_boot': int(n_boot)}

def min_attainable_p(n_pairs: int) -> float:
    if n_pairs <= 0:
        return float('nan')
    return float(min(1.0, 2.0 / 2.0 ** n_pairs))

def aggregate_seeds(per_seed: list[dict]) -> dict:
    d = [r['delta'] for r in per_seed]
    lo = [r['lo'] for r in per_seed]
    hi = [r['hi'] for r in per_seed]
    p = [r['p_two_sided'] for r in per_seed]
    direction = float(np.sign(np.mean(d)))
    agrees = [direction != 0.0 and np.sign(x) == direction for x in d]
    one_sided = [pi / 2.0 if ok else 1.0 - pi / 2.0 for pi, ok in zip(p, agrees)]
    return {'n_seeds': len(per_seed), 'seeds': ';'.join((str(r['seed']) for r in per_seed)), 'delta_mean': float(np.mean(d)), 'delta_min': float(np.min(d)), 'delta_max': float(np.max(d)), 'ci_union_lo': float(np.min(lo)), 'ci_union_hi': float(np.max(hi)), 'ci_lo_mean': float(np.mean(lo)), 'ci_hi_mean': float(np.mean(hi)), 'p_raw': float(min(1.0, 2.0 * max(one_sided))), 'p_max_seed': float(np.max(p)), 'sign_consistent': bool(all(agrees)), 'n_shared': int(min((r['n_shared'] for r in per_seed))), 'n_groups': int(min((r['n_groups'] for r in per_seed)))}

def _manifest() -> pd.DataFrame:
    return pd.read_parquet(require(MANIFEST, 'stats_rehab (unified manifest)'))

def disease_frame(cfg: str, seeds, man: pd.DataFrame) -> pd.DataFrame:
    return attach_groups(load_preds(cfg, seeds=tuple(seeds), kind='test'), man)

def biomass_individuals(df: pd.DataFrame, by_seed: bool=True) -> pd.DataFrame:
    d = df.copy()
    d['pred_weight_g'] = np.exp(d['pred_log_weight'].astype(float))
    keys = ['seed', 'individual_id'] if by_seed else ['individual_id']
    out = d.groupby(keys).agg(pred=('pred_weight_g', 'median'), y=('y_weight_g', 'first'), fold=('fold', 'first'), n_views=('pred_weight_g', 'count')).reset_index()
    return out

def _shared(ids_a, ids_b) -> list:
    return sorted(set(ids_a) & set(ids_b))

def disease_one_seed(da: pd.DataFrame, db: pd.DataFrame, n_boot: int, seed: int) -> dict:
    common = _shared(da['record_id'], db['record_id'])
    ia = da.set_index('record_id').loc[common]
    ib = db.set_index('record_id').loc[common]
    r = paired_bootstrap(ia['y_health'].values, ia['p_health'].values, ib['p_health'].values, ia['group_id'].values, balanced_accuracy, n_boot=n_boot, seed=seed)
    r['n_shared'] = len(common)
    r['folds_aligned'] = bool((ia['fold'].values == ib['fold'].values).all())
    return r

def biomass_one_seed(ta: pd.DataFrame, tb: pd.DataFrame, n_boot: int, seed: int) -> dict:
    common = _shared(ta['individual_id'], tb['individual_id'])
    ia = ta.set_index('individual_id').loc[common]
    ib = tb.set_index('individual_id').loc[common]
    r = paired_bootstrap(ia['y'].values, ia['pred'].values, ib['pred'].values, np.asarray(common), mae, n_boot=n_boot, seed=seed)
    r['n_shared'] = len(common)
    r['folds_aligned'] = bool((ia['fold'].values == ib['fold'].values).all())
    return r

def cross_seed_arrays(fa: pd.DataFrame, fb: pd.DataFrame, family: str, seeds) -> dict:
    id_col, y_col = ('record_id', 'y_health') if family == 'disease' else ('individual_id', 'y')
    pred_col = 'p_health' if family == 'disease' else 'pred'
    per_seed_ids = [set(fa[fa['seed'] == s][id_col]) & set(fb[fb['seed'] == s][id_col]) for s in seeds]
    common = sorted(set.intersection(*per_seed_ids))
    pa, pb, y_ref, groups = ([], [], None, None)
    for s in seeds:
        ia = fa[fa['seed'] == s].set_index(id_col).loc[common]
        ib = fb[fb['seed'] == s].set_index(id_col).loc[common]
        y = ia[y_col].values.astype(float)
        if not np.allclose(y, ib[y_col].values.astype(float), equal_nan=True):
            raise SystemExit(f'targets disagree between the two runs at seed {s}')
        if y_ref is None:
            y_ref = y
            groups = ia['group_id'].values if family == 'disease' else np.asarray(common)
        elif not np.allclose(y, y_ref, equal_nan=True):
            raise SystemExit(f'targets disagree between seeds at seed {s}')
        pa.append(ia[pred_col].values.astype(float))
        pb.append(ib[pred_col].values.astype(float))
    return {'y': y_ref, 'groups': groups, 'preds_a': pa, 'preds_b': pb, 'n_shared': len(common)}

def fold_seed_wilcoxon(fm: pd.DataFrame, a: str, b: str, family: str) -> dict:
    col = WILCOXON_METRIC[family]
    if col not in fm.columns:
        return {'p': None, 'n': 0, 'median_diff': None, 'note': f'{col} absent from fold_metrics.csv'}
    ga = fm[fm['config_name'] == a].set_index(['fold', 'seed'])
    gb = fm[fm['config_name'] == b].set_index(['fold', 'seed'])
    cells = [c for c in sorted(set(ga.index) & set(gb.index)) if np.isfinite(ga.loc[c, col]) and np.isfinite(gb.loc[c, col])]
    if not cells:
        return {'p': None, 'n': 0, 'median_diff': None, 'note': 'no (fold, seed) cell completed by both runs'}
    va = np.array([float(ga.loc[c, col]) for c in cells])
    vb = np.array([float(gb.loc[c, col]) for c in cells])
    w = wilcoxon_across_folds(va, vb)
    return {'p': None if not np.isfinite(w['pvalue']) else float(w['pvalue']), 'n': int(w['n']), 'median_diff': float(w['median_diff']), 'note': w.get('note', ''), 'metric': col}

def build_rows(seeds, n_boot: int, boot_seed: int, verbose: bool=True) -> tuple:
    man = _manifest()
    fm = pd.read_csv(require(METRICS / 'fold_metrics.csv', 'stats_rehab (fold metrics)'))
    fm = fm[fm['status'] == 'completed']
    dis_cache: dict[str, pd.DataFrame] = {}
    bio_cache: dict[str, pd.DataFrame] = {}

    def dis(cfg: str) -> pd.DataFrame:
        if cfg not in dis_cache:
            dis_cache[cfg] = disease_frame(cfg, seeds, man)
        return dis_cache[cfg]

    def bio(cfg: str) -> pd.DataFrame:
        if cfg not in bio_cache:
            bio_cache[cfg] = biomass_individuals(load_preds(cfg, seeds=tuple(seeds), kind='biomass_test'))
        return bio_cache[cfg]
    rows: list[dict] = []
    per_seed_meta: dict[str, list[dict]] = {}
    pooled_n: dict[str, int] = {}
    for family, grid in (('disease', DISEASE_COMPARISONS), ('biomass', BIOMASS_COMPARISONS)):
        for a, b, label in grid:
            fa, fb = (dis(a), dis(b)) if family == 'disease' else (bio(a), bio(b))
            shared_seeds = sorted(set(fa['seed']) & set(fb['seed']))
            per_seed = []
            for s in shared_seeds:
                sa, sb = (fa[fa['seed'] == s], fb[fb['seed'] == s])
                r = disease_one_seed(sa, sb, n_boot, boot_seed) if family == 'disease' else biomass_one_seed(sa, sb, n_boot, boot_seed)
                r['seed'] = int(s)
                per_seed.append(r)
            if not per_seed:
                raise SystemExit(f'no shared seed for {a} vs {b}')
            agg = aggregate_seeds(per_seed)
            w = fold_seed_wilcoxon(fm, a, b, family)
            arr = cross_seed_arrays(fa, fb, family, shared_seeds)
            pooled = pooled_seed_bootstrap(arr['y'], arr['preds_a'], arr['preds_b'], arr['groups'], balanced_accuracy if family == 'disease' else mae, n_boot=n_boot, seed=boot_seed)
            folds_aligned = all((r['folds_aligned'] for r in per_seed))
            w_valid = bool(folds_aligned and w['n'] > 0)
            if not w_valid and w['n'] > 0:
                w['note'] = "not comparable: the two runs' (fold, seed) cells score different held-out records, so this p compares different test populations -- render as n/a"
            p_floor = float(min(1.0, len(grid) / n_boot))
            key = (a, b)
            row = {'comparison': label, 'model_a': a, 'model_b': b, 'family': family, 'task': family, 'metric': METRIC_NAME[family], 'better': BETTER[family], 'tag': 'confirmatory' if key in CONFIRMATORY else 'exploratory', 'n_shared': agg['n_shared'], 'n_groups': agg['n_groups'], 'n_seeds': agg['n_seeds'], 'seeds': agg['seeds'], 'delta_mean': agg['delta_mean'], 'delta_min': agg['delta_min'], 'delta_max': agg['delta_max'], 'ci_pooled_lo': pooled['lo'], 'ci_pooled_hi': pooled['hi'], 'ci_union_lo': agg['ci_union_lo'], 'ci_union_hi': agg['ci_union_hi'], 'ci_lo_mean': agg['ci_lo_mean'], 'ci_hi_mean': agg['ci_hi_mean'], 'sign_consistent': agg['sign_consistent'], 'p_max_seed': agg['p_max_seed'], 'p_raw': agg['p_raw'], 'p_holm': float('nan'), 'p_floor': p_floor, 'p_holm_at_floor': False, 'holm_family_size': len(grid), 'wilcoxon_p': w['p'], 'wilcoxon_valid': w_valid, 'wilcoxon_n_pairs': w['n'], 'wilcoxon_min_attainable_p': min_attainable_p(w['n']), 'wilcoxon_median_diff': w['median_diff'], 'wilcoxon_note': w['note'], 'folds_aligned': folds_aligned}
            for s in DEFAULT_SEEDS:
                hit = next((r for r in per_seed if r['seed'] == s), None)
                row[f'delta_s{s}'] = hit['delta'] if hit else None
                row[f'p_s{s}'] = hit['p_two_sided'] if hit else None
            rows.append(row)
            per_seed_meta[f'{a} vs {b}'] = [{'seed': r['seed'], 'delta': r['delta'], 'lo': r['lo'], 'hi': r['hi'], 'p_two_sided': r['p_two_sided'], 'n_shared': r['n_shared'], 'n_groups': r['n_groups'], 'metric_a': r['metric_a'], 'metric_b': r['metric_b']} for r in per_seed]
            pooled_n[f'{a} vs {b}'] = arr['n_shared']
            if verbose:
                print(f"  {family:8s} {a} vs {b}: delta {agg['delta_mean']:+.4f} pooled 95 % [{pooled['lo']:+.4f}, {pooled['hi']:+.4f}] union [{agg['ci_union_lo']:+.4f}, {agg['ci_union_hi']:+.4f}]  p_raw {agg['p_raw']:.4f}  n={agg['n_shared']} ({agg['n_groups']} clusters) seeds {agg['seeds']}")
    for family in ('disease', 'biomass'):
        idx = [i for i, r in enumerate(rows) if r['family'] == family]
        adj = holm([rows[i]['p_raw'] for i in idx])
        for i, v in zip(idx, adj):
            rows[i]['p_holm'] = v
            rows[i]['p_holm_at_floor'] = bool(v <= rows[i]['p_floor'] + 1e-12)
    return (rows, per_seed_meta, pooled_n)

def registry_summary() -> dict:
    p = METRICS / 'experiment_registry.csv'
    if not p.exists():
        return {'rows': None, 'seeds': [], 'note': 'experiment_registry.csv absent'}
    reg = pd.read_csv(p)
    done = reg[reg['status'] == 'completed']
    return {'rows': int(len(reg)), 'completed_rows': int(len(done)), 'seeds': sorted((int(s) for s in done['seed'].unique())), 'configs': int(done['run_name'].str.rsplit('_f', n=1).str[0].nunique())}

def _evaluate_style_pooled(a: str, b: str, family: str, seeds, man, n_boot: int, boot_seed: int) -> dict:
    if family == 'disease':
        da, db = (disease_frame(a, seeds, man), disease_frame(b, seeds, man))
        common = _shared(da['record_id'], db['record_id'])
        ia = da.set_index('record_id').loc[common]
        ib = db.set_index('record_id').loc[common]
        r = paired_bootstrap(ia['y_health'].values, ia['p_health'].values, ib['p_health'].values, ia['group_id'].values, balanced_accuracy, n_boot=n_boot, seed=boot_seed)
    else:
        ta = biomass_individuals(load_preds(a, seeds=tuple(seeds), kind='biomass_test'), by_seed=False)
        tb = biomass_individuals(load_preds(b, seeds=tuple(seeds), kind='biomass_test'), by_seed=False)
        common = _shared(ta['individual_id'], tb['individual_id'])
        ia = ta.set_index('individual_id').loc[common]
        ib = tb.set_index('individual_id').loc[common]
        r = paired_bootstrap(ia['y'].values, ia['pred'].values, ib['pred'].values, np.asarray(common), mae, n_boot=n_boot, seed=boot_seed)
    r['n_shared'] = len(common)
    return r

def check_against(report_path: Path, seeds, n_boot: int, boot_seed: int, tol: float=0.0001) -> int:
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    ref = {(c['model_a'], c['model_b']): c for c in report.get('paired_comparisons', []) if c.get('status') == 'ok'}
    man = _manifest()
    mtime = pd.Timestamp(Path(report_path).stat().st_mtime, unit='s', tz='UTC')
    print(f"check against {rel(report_path)}  (mtime {mtime.tz_convert(None)} UTC, n_boot {report.get('n_boot')})")
    print(f'this script: n_boot {n_boot}, bootstrap seed {boot_seed}, seeds {list(seeds)}, tolerance {tol}')
    print('')
    hdr = f"{'comparison':64s} {'d_eval':>9s} {'d_pool':>9s} {'|dd|':>8s} {'|dlo|':>8s} {'|dhi|':>8s} {'d_smean':>9s} {'|dd|sm':>8s} {'d_s42':>9s} {'|dlo|cp':>8s} {'|dhi|cp':>8s}"
    print(hdr)
    print('-' * len(hdr))
    failures, checked = ([], 0)
    for family, grid in (('disease', DISEASE_COMPARISONS), ('biomass', BIOMASS_COMPARISONS)):
        for a, b, label in grid:
            c = ref.get((a, b))
            if c is None:
                print(f"{a + ' vs ' + b:64s} {'absent from the report':>9s}")
                continue
            pooled = _evaluate_style_pooled(a, b, family, seeds, man, n_boot, boot_seed)
            fa, fb = (disease_frame(a, seeds, man), disease_frame(b, seeds, man)) if family == 'disease' else (biomass_individuals(load_preds(a, seeds=tuple(seeds), kind='biomass_test')), biomass_individuals(load_preds(b, seeds=tuple(seeds), kind='biomass_test')))
            per_seed = []
            for s in sorted(set(fa['seed']) & set(fb['seed'])):
                sa, sb = (fa[fa['seed'] == s], fb[fb['seed'] == s])
                r = disease_one_seed(sa, sb, n_boot, boot_seed) if family == 'disease' else biomass_one_seed(sa, sb, n_boot, boot_seed)
                r['seed'] = int(s)
                per_seed.append(r)
            agg = aggregate_seeds(per_seed)
            s42 = next((r['delta'] for r in per_seed if r['seed'] == 42), float('nan'))
            arr = cross_seed_arrays(fa, fb, family, sorted(set(fa['seed']) & set(fb['seed'])))
            cp = pooled_seed_bootstrap(arr['y'], arr['preds_a'], arr['preds_b'], arr['groups'], balanced_accuracy if family == 'disease' else mae, n_boot=n_boot, seed=boot_seed)
            dd = abs(pooled['delta'] - c['diff'])
            dlo = abs(pooled['lo'] - c['lo'])
            dhi = abs(pooled['hi'] - c['hi'])
            dsm = abs(agg['delta_mean'] - c['diff'])
            cplo = abs(cp['lo'] - c['lo'])
            cphi = abs(cp['hi'] - c['hi'])
            print(f"{a + ' vs ' + b:64s} {c['diff']:9.5f} {pooled['delta']:9.5f} {dd:8.2e} {dlo:8.2e} {dhi:8.2e} {agg['delta_mean']:9.5f} {dsm:8.2e} {s42:9.5f} {cplo:8.2e} {cphi:8.2e}")
            checked += 1
            for what, v in (('delta', dd), ('ci_lo', dlo), ('ci_hi', dhi)):
                if not v <= tol:
                    failures.append(f'{a} vs {b}: pooled {what} differs by {v:.2e}')
            if family == 'disease':
                if not dsm <= tol:
                    failures.append(f'{a} vs {b}: seed-mean delta differs by {dsm:.2e}')
                for what, v in (('ci_pooled_lo', cplo), ('ci_pooled_hi', cphi)):
                    if not v <= tol:
                        failures.append(f'{a} vs {b}: {what} differs by {v:.2e}')
    print('')
    if failures:
        print(f'AGREEMENT FAILED on {len(failures)} check(s):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print(f'AGREEMENT OK: {checked} comparisons agree with {rel(report_path)} to {tol:g}Numerical result computed from approved input artifacts.')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=BOOT_SEED, help='bootstrap RNG seed (not a training seed)')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='training seeds to aggregate over')
    ap.add_argument('--check-against', type=Path, default=None, help='evaluation_report.json to cross-check against; writes nothing')
    ap.add_argument('--tol', type=float, default=0.0001)
    args = ap.parse_args()
    if args.check_against is not None:
        return check_against(require(args.check_against, 'stats_rehab --check-against'), args.seeds, args.n_boot, args.seed, args.tol)
    reg = registry_summary()
    print(f"registry: {reg.get('completed_rows')} completed rows, seeds {reg['seeds']}")
    rows, per_seed, pooled_n = build_rows(args.seeds, args.n_boot, args.seed)
    df = pd.DataFrame(rows)

    def _flips(fam=None):
        return [r for r in rows if not r['sign_consistent'] and (fam is None or r['family'] == fam)]

    def _n_sig_seeds(r):
        return sum((1 for s in per_seed[f"{r['model_a']} vs {r['model_b']}"] if s['p_two_sided'] < 0.05))

    def _all_ci_clear(r):
        return all((s['lo'] > 0 or s['hi'] < 0 for s in per_seed[f"{r['model_a']} vs {r['model_b']}"]))
    pooled_clear_but_not_significant = [r['comparison'] for r in rows if r['p_holm'] > 0.05 and (r['ci_pooled_lo'] > 0 or r['ci_pooled_hi'] < 0)]
    flips = _flips()
    flips_all_ci = [r['comparison'] for r in flips if _all_ci_clear(r)]
    flips_2sig = [r for r in flips if _n_sig_seeds(r) >= 2]
    at_floor = [r['comparison'] for r in rows if r['wilcoxon_p'] is not None and abs(r['wilcoxon_p'] - r['wilcoxon_min_attainable_p']) < 1e-15]
    survivors = [r['comparison'] for r in rows if r['p_holm'] <= 0.05]
    seed_reversals = {'n_rows': len(rows), 'n_sign_flipping': len(flips), 'n_sign_flipping_by_family': {f: len(_flips(f)) for f in ('disease', 'biomass')}, 'n_flipping_with_all_per_seed_intervals_clear_of_zero': len(flips_all_ci), 'flipping_with_all_per_seed_intervals_clear_of_zero': flips_all_ci, 'n_flipping_with_two_or_more_significant_seeds': len(flips_2sig), 'n_flipping_with_two_or_more_significant_seeds_by_family': {f: sum((1 for r in flips_2sig if r['family'] == f)) for f in ('disease', 'biomass')}, 'note': 'Numerical result computed from approved input artifacts.'}
    n_pairs = sorted({int(r['wilcoxon_n_pairs']) for r in rows if r['wilcoxon_n_pairs']})
    meta = {'what': 'Paired cluster comparisons over training seeds', 'n_boot': args.n_boot, 'alpha': 0.05, 'bootstrap_seed': args.seed, 'seeds': [int(s) for s in args.seeds], 'registry': reg, 'metrics': {'disease': 'health balanced accuracy at 0.50', 'biomass': 'individual weight MAE in grams'}, 'pairing': 'shared records at every seed; fused-group disease clusters and individual biomass clusters', 'holm_families': ['disease', 'biomass'], 'definitions': {'raw_p': 'min(1, 2 * max_seed directional_one_sided_p)', 'ci_union': 'envelope of per-seed percentile intervals', 'ci_pooled': 'paired bootstrap of mean seed delta; conditional on trained models', 'p_floor': '1 / n_boot per seed', 'holm': 'step-down running maximum within task family', 'delta': 'metric(model_a) - metric(model_b)'}, 'wilcoxon': {'metric': WILCOXON_METRIC, 'n_pairs_present': n_pairs, 'min_attainable_p_by_n': {str(n): min_attainable_p(n) for n in n_pairs}, 'n_rows_not_comparable': sum((1 for r in rows if not r['wilcoxon_valid'])), 'rows_at_their_floor': at_floor}, 'seed_reversals': seed_reversals, 'pooled_ci': {'columns': ['ci_pooled_lo', 'ci_pooled_hi'], 'n_shared': pooled_n, 'n_rows_clear_of_zero_with_p_holm_above_0.05': len(pooled_clear_but_not_significant), 'rows_clear_of_zero_with_p_holm_above_0.05': pooled_clear_but_not_significant}, 'survivors_at_0.05': survivors, 'sources': ['outputs/predictions/<config>_f<fold>_s<seed>_test.parquet', 'outputs/predictions/<config>_f<fold>_s<seed>_biomass_test.parquet', 'outputs/metrics/fold_metrics.csv', 'outputs/metrics/experiment_registry.csv', 'data/manifests/unified_manifest.parquet'], 'per_seed': per_seed}
    write_table('t_ablation_stats', df, meta)
    alpha = 0.05
    for family in ('disease', 'biomass'):
        sub = df[df['family'] == family]
        surv = sub[sub['p_holm'] <= alpha]
        print(f'\n{family}: {len(surv)}/{len(sub)} comparisons survive Holm at {alpha}')
        for _, r in surv.iterrows():
            print(f"  {r['comparison']}: delta {r['delta_mean']:+.4f} Holm p {r['p_holm']:.4f} ({r['tag']})")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
