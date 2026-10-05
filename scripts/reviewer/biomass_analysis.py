"""Compute individual-level biomass metrics across database and site protocols."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MANIFEST, METRICS, PRED, SPLITS, load_preds, rel, require, write_table
from src.evaluation.biomass_baselines import individual_table
from src.evaluation.bootstrap import grouped_bootstrap_ci
from src.evaluation.metrics import biomass_metrics
from stats_rehab import cross_seed_arrays, pooled_seed_bootstrap
SEED = 42
PRED_SEEDS = (42, 2026, 3407)
N_BOOT = 2000
MAPE_FLOOR_G = 0.5
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1)
MIN_RF_TRAIN = 5
MIN_ALLO_TRAIN = 3
COLUMNS = ['protocol', 'held_out_unit', 'n_individuals', 'n_images', 'image_mae_g', 'image_mae_ci_lo', 'image_mae_ci_hi', 'image_mae_seed_sd_g', 'image_r2', 'image_r2_centred', 'train_mean_mae_g', 'train_mean_r2', 'db_mean_mae_g', 'db_mean_r2', 'rf_length_mae_g', 'rf_length_r2', 'rf_length_only_mae_g', 'rf_length_only_r2', 'allometric_mae_g', 'allometric_r2', 'train_weight_min_g', 'train_weight_max_g', 'test_weight_min_g', 'test_weight_max_g', 'test_weight_sd_g', 'rf_features', 'rf_train_n', 'rf_train_weight_min_g', 'rf_train_weight_max_g', 'rf_length_only_train_n', 'reference_source', 'n_seeds']
TIER_COLUMNS = ['held_out_db', 'n_test', 'n_train', 'mae_rf_length', 'r2_rf_length', 'mae_rf_length_plus_image', 'r2_rf_length_plus_image', 'delta_mae']
PROTOCOLS = [{'key': 'LODO', 'cfg': 'E4_biomass_single_image', 'split': 'biomass_lodo', 'unit': 'fold', 'folds': 5, 'csv': 'biomass_baselines.csv', 'label': 'leave-one-database-out (5 folds)'}, {'key': 'LOSO', 'cfg': 'E12_biomass_loso', 'split': 'biomass_loso', 'unit': 'fold', 'folds': 3, 'csv': 'biomass_baselines_loso.csv', 'label': 'leave-one-site-out (3 folds; five databases are three sites)'}, {'key': 'within_DB', 'cfg': 'E13_biomass_within_db', 'split': 'biomass_within_db', 'unit': 'db', 'folds': 5, 'csv': 'biomass_baselines_within_db.csv', 'label': 'within-database grouped CV (5 folds, reported per database)'}]

def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))

def r2_centred(y, p, db) -> float:
    """R2 using the sum of within-database target variances as denominator."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    db = np.asarray(db)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p, db = (y[keep], p[keep], db[keep])
    if y.size < 2:
        return float('nan')
    ss_res = float(np.sum((p - y) ** 2))
    ss_tot = 0.0
    for d in np.unique(db):
        m = db == d
        ss_tot += float(np.sum((y[m] - y[m].mean()) ** 2))
    if ss_tot <= 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot

def individuals_from_preds(df: pd.DataFrame) -> pd.DataFrame:
    """Take each animal's median view prediction after exponentiating weight."""
    d = df.copy()
    d['pred_weight_g'] = np.exp(d['pred_log_weight'].astype(float))
    out = d.groupby('individual_id').agg(pred_weight_g=('pred_weight_g', 'median'), pred_log_weight=('pred_log_weight', 'median'), pred_length_cm=('pred_length_cm', 'median'), y_weight_g=('y_weight_g', 'first'), y_length_cm=('y_length_cm', 'first'), db=('db', 'first'), fold=('fold', 'first'), n_views=('pred_weight_g', 'count')).reset_index()
    return out

def split_individuals(splits: pd.DataFrame, manifest: pd.DataFrame, protocol: str) -> pd.DataFrame:
    ids = manifest.set_index('record_id')['individual_id']
    s = splits[splits['protocol'] == protocol].copy()
    if s.empty:
        raise SystemExit(f'protocol {protocol} absent from {rel(SPLITS)}')
    s['individual_id'] = s['record_id'].map(ids)
    s = s.dropna(subset=['individual_id'])
    return s[['individual_id', 'fold', 'split']].drop_duplicates().reset_index(drop=True)

def rf_feature_names(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    feats = ['length_cm']
    if train['cephalothorax_cm'].notna().sum() > 3 and test['cephalothorax_cm'].notna().sum() > 0:
        feats.append('cephalothorax_cm')
    return feats

def rf_length_reference(train: pd.DataFrame, test: pd.DataFrame, seed: int=SEED, feats: list[str] | None=None) -> dict:
    feats = list(feats) if feats else rf_feature_names(train, test)
    tr = train.dropna(subset=feats + ['weight_g'])
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    info = {'features': '+'.join(feats), 'n_train': int(len(tr)), 'train_weight_min_g': float(tr['weight_g'].min()) if len(tr) else float('nan'), 'train_weight_max_g': float(tr['weight_g'].max()) if len(tr) else float('nan')}
    te_ok = test.dropna(subset=feats)
    if len(tr) >= MIN_RF_TRAIN and len(te_ok) >= 1:
        mdl = RandomForestRegressor(random_state=seed, **RF_KW)
        mdl.fit(tr[feats].values, tr['weight_g'].values)
        pred.loc[te_ok.index] = mdl.predict(te_ok[feats].values)
    info['pred'] = pred
    return info

def allometric_reference(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr = train.dropna(subset=['weight_g', 'length_cm'])
    tr = tr[(tr['weight_g'] > 0) & (tr['length_cm'] > 0)]
    pred = pd.Series(np.nan, index=test.index, dtype=float)
    info = {'a': float('nan'), 'b': float('nan'), 'n_train': int(len(tr))}
    if len(tr) < MIN_ALLO_TRAIN:
        info['pred'] = pred
        return info
    b, log_a = np.polyfit(np.log(tr['length_cm'].values), np.log(tr['weight_g'].values), 1)
    te = test[test['length_cm'].notna() & (test['length_cm'] > 0)]
    pred.loc[te.index] = np.exp(log_a + b * np.log(te['length_cm'].values))
    info.update({'a': float(np.exp(log_a)), 'b': float(b), 'pred': pred})
    return info

def allometric_on_forest_subset(ind: pd.DataFrame, split_ind: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in sorted(split_ind['fold'].unique()):
        f = split_ind[split_ind['fold'] == fold]
        tr = ind[ind['individual_id'].isin(set(f[f['split'] == 'train']['individual_id']))]
        te = ind[ind['individual_id'].isin(set(f[f['split'] == 'test']['individual_id']))]
        if te.empty:
            continue
        feats = rf_feature_names(tr, te)
        sub = tr.dropna(subset=feats + ['weight_g'])
        a = allometric_reference(sub, te)
        m = biomass_metrics(te['weight_g'].values.astype(float), a['pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
        rows.append({'held_out_unit': ';'.join(sorted(te['pond_or_db_id'].astype(str).unique())), 'n_train': a['n_train'], 'b': round(a['b'], 4), 'mae_g': round(m['mae'], 4), 'r2': round(m['r2'], 4)})
    return pd.DataFrame(rows)

def train_mean_reference(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), float(train['weight_g'].mean()))

def db_mean_reference(test: pd.DataFrame) -> np.ndarray:
    return test.groupby('db')['y_weight_g'].transform('mean').values.astype(float)

def reference_frame(ind: pd.DataFrame, split_ind: pd.DataFrame, seed: int=SEED, unit: str='fold') -> pd.DataFrame:
    if unit not in ('fold', 'db'):
        raise SystemExit(f"unit must be 'fold' or 'db', got {unit!r}")
    rows = []
    for fold in sorted(split_ind['fold'].unique()):
        f = split_ind[split_ind['fold'] == fold]
        tr_ids = set(f[f['split'] == 'train']['individual_id'])
        te_ids = set(f[f['split'] == 'test']['individual_id'])
        tr = ind[ind['individual_id'].isin(tr_ids)]
        te_all = ind[ind['individual_id'].isin(te_ids)]
        if te_all.empty:
            continue
        held = ';'.join(sorted(te_all['pond_or_db_id'].astype(str).unique()))
        groups = [('', te_all)] if unit == 'fold' else [(str(d), g) for d, g in te_all.groupby('pond_or_db_id')]
        for _, te in groups:
            rf = rf_length_reference(tr, te, seed=seed)
            rf_len = rf_length_reference(tr, te, seed=seed, feats=['length_cm'])
            allo = allometric_reference(tr, te)
            rows.append(pd.DataFrame({'individual_id': te['individual_id'].values, 'db': te['pond_or_db_id'].astype(str).values, 'fold': int(fold), 'held_out_dbs': held, 'y_weight_g': te['weight_g'].values.astype(float), 'y_length_cm': te['length_cm'].values.astype(float), 'train_mean_pred': train_mean_reference(tr, te), 'rf_pred': rf['pred'].values, 'rf_length_only_pred': rf_len['pred'].values, 'allometric_pred': allo['pred'].values, 'n_train': int(len(tr)), 'train_weight_min_g': float(tr['weight_g'].min()), 'train_weight_max_g': float(tr['weight_g'].max()), 'rf_features': rf['features'], 'rf_train_n': rf['n_train'], 'rf_train_weight_min_g': rf['train_weight_min_g'], 'rf_train_weight_max_g': rf['train_weight_max_g'], 'rf_length_only_train_n': rf_len['n_train'], 'allometric_a': allo['a'], 'allometric_b': allo['b']}))
    if not rows:
        raise SystemExit('no held-out individuals found for this protocol')
    return pd.concat(rows, ignore_index=True)

def out_of_fold_features(per_fold: dict) -> pd.DataFrame:
    frames = []
    for fold in sorted(per_fold):
        g = per_fold[fold].groupby('individual_id').agg(f_pred_log_weight=('pred_log_weight', 'median'), f_pred_length_cm=('pred_length_cm', 'median')).reset_index()
        g['feature_fold'] = int(fold)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    dup = out['individual_id'][out['individual_id'].duplicated()].unique()
    if len(dup):
        raise SystemExit(f'{len(dup)} individuals are test animals in more than one fold (e.g. {list(dup[:3])}): the out-of-fold feature join is ambiguous')
    return out[['individual_id', 'feature_fold', 'f_pred_log_weight', 'f_pred_length_cm']]

def check_no_feature_leak(split_ind: pd.DataFrame, oof: pd.DataFrame) -> None:
    """Require every image feature to come from a fold holding its animal out."""
    train_pairs = set(zip(split_ind[split_ind['split'] == 'train']['individual_id'], split_ind[split_ind['split'] == 'train']['fold'].astype(int)))
    test_pairs = set(zip(split_ind[split_ind['split'] == 'test']['individual_id'], split_ind[split_ind['split'] == 'test']['fold'].astype(int)))
    pairs = list(zip(oof['individual_id'], oof['feature_fold'].astype(int)))
    leaked = [p for p in pairs if p in train_pairs]
    if leaked:
        raise SystemExit(f'out-of-fold join leaks: {len(leaked)} individuals would take image features from a fold that TRAINED on them (e.g. {leaked[:3]})')
    stranger = [p for p in pairs if p not in test_pairs]
    if stranger:
        raise SystemExit(f'out-of-fold join is not out-of-fold: {len(stranger)} individuals take features from a fold in which they are not a test animal (e.g. {stranger[:3]})')

def _rf_fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], seed: int) -> np.ndarray:
    tr = tr.dropna(subset=feats + ['weight_g'])
    out = np.full(len(te), np.nan)
    te_ok = te.dropna(subset=feats)
    if len(tr) < MIN_RF_TRAIN or len(te_ok) < 1:
        return out
    mdl = RandomForestRegressor(random_state=seed, **RF_KW)
    mdl.fit(tr[feats].values, tr['weight_g'].values)
    out[[te.index.get_loc(i) for i in te_ok.index]] = mdl.predict(te_ok[feats].values)
    return out

def tabular_tier_rows(ind: pd.DataFrame, split_ind: pd.DataFrame, oof: pd.DataFrame, seed: int=SEED) -> pd.DataFrame:
    base = ['length_cm']
    plus = ['length_cm', 'f_pred_log_weight', 'f_pred_length_cm']
    tab = ind.merge(oof, on='individual_id', how='left')
    rows = []
    for fold in sorted(split_ind['fold'].unique()):
        f = split_ind[split_ind['fold'] == fold]
        tr = tab[tab['individual_id'].isin(set(f[f['split'] == 'train']['individual_id']))]
        te = tab[tab['individual_id'].isin(set(f[f['split'] == 'test']['individual_id']))]
        if te.empty:
            continue
        tr, te = (tr.reset_index(drop=True), te.reset_index(drop=True))
        y = te['weight_g'].values.astype(float)
        p_base = _rf_fit_predict(tr, te, base, seed)
        p_plus = _rf_fit_predict(tr, te, plus, seed)
        m_base = biomass_metrics(y, p_base, unit='g', mape_floor=MAPE_FLOOR_G)
        m_plus = biomass_metrics(y, p_plus, unit='g', mape_floor=MAPE_FLOOR_G)
        rows.append({'held_out_db': ';'.join(sorted(te['pond_or_db_id'].astype(str).unique())), 'n_test': int(len(te)), 'n_train': int(len(tr)), 'mae_rf_length': m_base['mae'], 'r2_rf_length': m_base['r2'], 'mae_rf_length_plus_image': m_plus['mae'], 'r2_rf_length_plus_image': m_plus['r2'], 'delta_mae': m_plus['mae'] - m_base['mae']})
    return pd.DataFrame(rows, columns=TIER_COLUMNS)

def _csv_reference(csv: pd.DataFrame, unit: str, model: str) -> tuple[float, float]:
    m = csv[(csv['held_out_db'] == unit) & csv['model'].str.startswith(model)]
    if len(m) != 1:
        return (float('nan'), float('nan'))
    return (float(m['weight_mae'].iloc[0]), float(m['weight_r2'].iloc[0]))

def _seed_frames(spec: dict, img: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {}
    for s in sorted(img['seed'].unique()):
        one = img[img['seed'] == s]
        d = ref.merge(one[['individual_id', 'pred_weight_g', 'n_views']], on='individual_id', how='inner', validate='one_to_one')
        if len(d) != len(ref):
            raise SystemExit(f"{spec['cfg']} seed {s}: {len(ref)} held-out individuals in the split table but {len(d)} matched prediction rows")
        d['unit'] = d['held_out_dbs'] if spec['unit'] == 'fold' else d['db']
        out[int(s)] = d
    if not out:
        raise SystemExit(f"{spec['cfg']}: no prediction seeds loaded")
    return out

def _seed_mean(values) -> float:
    return float(np.mean([v for v in values]))

def _seed_sd(values) -> float:
    vals = [float(v) for v in values]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan')

def protocol_rows(spec: dict, img: pd.DataFrame, ref: pd.DataFrame, csv: pd.DataFrame, n_images: dict, n_boot: int=N_BOOT, seed: int=SEED) -> tuple[pd.DataFrame, list[dict]]:
    frames = _seed_frames(spec, img, ref)
    seeds = sorted(frames)
    rows, per_seed = ([], [])
    for unit in sorted(set(frames[seeds[0]]['unit'].astype(str))):
        groups = {s: d[d['unit'].astype(str) == unit] for s, d in frames.items()}
        g = groups[seeds[0]]
        y = g['y_weight_g'].values.astype(float)
        maes, r2s, r2cs = ([], [], [])
        for s in seeds:
            gs = groups[s]
            ps = gs['pred_weight_g'].values.astype(float)
            ms = biomass_metrics(gs['y_weight_g'].values.astype(float), ps, unit='g', mape_floor=MAPE_FLOOR_G)
            r2c = r2_centred(gs['y_weight_g'].values.astype(float), ps, gs['db'].values) if spec['key'] == 'within_DB' else float('nan')
            maes.append(ms['mae'])
            r2s.append(ms['r2'])
            r2cs.append(r2c)
            per_seed.append({'protocol': spec['key'], 'held_out_unit': unit, 'seed': int(s), 'n_individuals': int(len(gs)), 'image_mae_g': ms['mae'], 'image_r2': ms['r2'], 'image_r2_centred': r2c})
        stacked = pd.concat([groups[s] for s in seeds], ignore_index=True)
        ci = grouped_bootstrap_ci(_mae, stacked['y_weight_g'].values.astype(float), stacked['pred_weight_g'].values.astype(float), stacked['individual_id'].values, n_boot=n_boot, seed=seed)
        img_m = {'mae': _seed_mean(maes), 'r2': _seed_mean(r2s)}
        tm = biomass_metrics(y, g['train_mean_pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
        dbm = biomass_metrics(y, db_mean_reference(g), unit='g', mape_floor=MAPE_FLOOR_G)
        rfl = biomass_metrics(y, g['rf_length_only_pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
        if spec['unit'] == 'fold':
            rf_mae, rf_r2 = _csv_reference(csv, str(unit), 'random_forest')
            al_mae, al_r2 = _csv_reference(csv, str(unit), 'allometric')
            source = f"outputs/metrics/{spec['csv']}"
        else:
            rf = biomass_metrics(y, g['rf_pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
            al = biomass_metrics(y, g['allometric_pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
            rf_mae, rf_r2, al_mae, al_r2 = (rf['mae'], rf['r2'], al['mae'], al['r2'])
            source = 'refitted here per database: the within-database CSV is per fold and pooled over all five databases, so it carries no per-database row'
        feats = sorted(set(g['rf_features']))
        rows.append({'protocol': spec['key'], 'held_out_unit': str(unit), 'n_individuals': int(len(g)), 'n_images': int(n_images.get(str(unit), 0)), 'image_mae_g': img_m['mae'], 'image_mae_ci_lo': ci['lo'], 'image_mae_ci_hi': ci['hi'], 'image_mae_seed_sd_g': _seed_sd(maes), 'image_r2': img_m['r2'], 'image_r2_centred': _seed_mean(r2cs) if spec['key'] == 'within_DB' else float('nan'), 'train_mean_mae_g': tm['mae'], 'train_mean_r2': tm['r2'], 'db_mean_mae_g': dbm['mae'], 'db_mean_r2': dbm['r2'], 'rf_length_mae_g': rf_mae, 'rf_length_r2': rf_r2, 'rf_length_only_mae_g': rfl['mae'], 'rf_length_only_r2': rfl['r2'], 'allometric_mae_g': al_mae, 'allometric_r2': al_r2, 'train_weight_min_g': float(g['train_weight_min_g'].min()), 'train_weight_max_g': float(g['train_weight_max_g'].max()), 'test_weight_min_g': float(np.min(y)), 'test_weight_max_g': float(np.max(y)), 'test_weight_sd_g': float(np.std(y, ddof=1)) if len(y) > 1 else float('nan'), 'rf_features': feats[0] if len(feats) == 1 else 'mixed: ' + ', '.join(feats), 'rf_train_n': int(g['rf_train_n'].min()), 'rf_train_weight_min_g': float(g['rf_train_weight_min_g'].min()), 'rf_train_weight_max_g': float(g['rf_train_weight_max_g'].max()), 'rf_length_only_train_n': int(g['rf_length_only_train_n'].min()), 'reference_source': source, 'n_seeds': len(seeds)})
    return (pd.DataFrame(rows, columns=COLUMNS), per_seed)

def pooled_line(spec: dict, img: pd.DataFrame, ref: pd.DataFrame) -> dict:
    frames = _seed_frames(spec, img, ref)
    seeds = sorted(frames)
    per_seed = []
    for s in seeds:
        d = frames[s]
        y = d['y_weight_g'].values.astype(float)
        p = d['pred_weight_g'].values.astype(float)
        m = biomass_metrics(y, p, unit='g', mape_floor=MAPE_FLOOR_G)
        per_seed.append({'seed': int(s), 'image_mae_g': round(m['mae'], 4), 'image_r2': round(m['r2'], 4), 'image_r2_centred': round(r2_centred(y, p, d['db'].values), 4)})
    d = frames[seeds[0]]
    y = d['y_weight_g'].values.astype(float)
    tm = biomass_metrics(y, d['train_mean_pred'].values, unit='g', mape_floor=MAPE_FLOOR_G)
    dbm = biomass_metrics(y, db_mean_reference(d), unit='g', mape_floor=MAPE_FLOOR_G)
    return {'protocol': spec['key'], 'n_individuals': int(len(d)), 'n_seeds': len(seeds), 'image_mae_g': round(_seed_mean([r['image_mae_g'] for r in per_seed]), 4), 'image_mae_seed_sd_g': round(_seed_sd([r['image_mae_g'] for r in per_seed]), 4), 'image_r2': round(_seed_mean([r['image_r2'] for r in per_seed]), 4), 'image_r2_centred': round(_seed_mean([r['image_r2_centred'] for r in per_seed]), 4), 'train_mean_mae_g': round(tm['mae'], 4), 'train_mean_r2': round(tm['r2'], 4), 'db_mean_oracle_mae_g': round(dbm['mae'], 4), 'db_mean_oracle_r2': round(dbm['r2'], 4), 'per_seed': per_seed}
CROSS_SITE_REFERENCES = {'rf_length_only_mae_g': 'the forest on length alone', 'allometric_mae_g': 'the parametric law ln W = a + b ln L', 'rf_length_mae_g': 'the forest as run under the repo feature rule'}
POOLED_META_KEY = 'pooled_reference'

def cross_site_reference_ranking(table: pd.DataFrame) -> list[tuple[str, float]]:
    means = [(name, float(table[col].mean())) for col, name in CROSS_SITE_REFERENCES.items()]
    return sorted(means, key=lambda kv: kv[1])

def pooled_reference(rows: list[dict]) -> dict:
    return {'rows': rows}

def sd_collapse(img: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for db, g in img.groupby('db'):
        animals = g.drop_duplicates('individual_id')
        per_seed = [s for _k, s in g.groupby('seed')]
        rows.append({'db': str(db), 'n': int(len(animals)), 'n_seeds': int(g['seed'].nunique()), 'true_weight_sd_g': float(np.std(animals['y_weight_g'], ddof=1)), 'pred_weight_sd_g': float(np.mean([np.std(s['pred_weight_g'], ddof=1) for s in per_seed])), 'true_length_sd_cm': float(np.std(animals['y_length_cm'].dropna(), ddof=1)), 'pred_length_sd_cm': float(np.mean([np.std(s['pred_length_cm'], ddof=1) for s in per_seed]))})
    return pd.DataFrame(rows)

def load_protocol_predictions(spec: dict, seeds=PRED_SEEDS) -> tuple:
    seeds = tuple((int(s) for s in seeds))
    for s in seeds:
        for f in range(spec['folds']):
            require(PRED / f"{spec['cfg']}_f{f}_s{s}_biomass_test.parquet", f"{spec['key']} fold {f} seed {s}")
    preds = load_preds(spec['cfg'], seeds=seeds, kind='biomass_test')
    for s in seeds:
        got = sorted(preds[preds['seed'] == s]['fold'].unique())
        if len(got) != spec['folds']:
            raise SystemExit(f"{spec['cfg']} seed {s}: expected {spec['folds']} folds, found {got}")
    img = pd.concat([individuals_from_preds(preds[preds['seed'] == s]).assign(seed=s) for s in seeds], ignore_index=True).assign(n_seeds=len(seeds))
    return (preds, img)

def _stats_frame(img: pd.DataFrame) -> pd.DataFrame:
    return img.rename(columns={'pred_weight_g': 'pred', 'y_weight_g': 'y'})[['seed', 'individual_id', 'pred', 'y', 'fold']]

def seed_mean_individuals(img: pd.DataFrame) -> pd.DataFrame:
    return img.groupby('individual_id').agg(pred_weight_g=('pred_weight_g', 'mean'), pred_length_cm=('pred_length_cm', 'mean'), y_weight_g=('y_weight_g', 'first'), y_length_cm=('y_length_cm', 'first'), db=('db', 'first'), fold=('fold', 'first'), n_seeds=('seed', 'nunique')).reset_index()

def load_lodo_predictions(seeds=PRED_SEEDS) -> pd.DataFrame:
    spec = next((s for s in PROTOCOLS if s['key'] == 'LODO'))
    return load_protocol_predictions(spec, seeds=seeds)[1]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--seed', type=int, default=SEED, help='random-forest and cluster-bootstrap seed (not a run)')
    ap.add_argument('--pred-seeds', default=','.join((str(s) for s in PRED_SEEDS)), help='prediction seeds every metric is averaged over')
    args = ap.parse_args()
    pred_seeds = tuple((int(s) for s in str(args.pred_seeds).split(',') if s.strip()))
    require(MANIFEST, 'unified manifest')
    require(SPLITS, 'split table')
    man = pd.read_parquet(MANIFEST)
    splits = pd.read_parquet(SPLITS)
    ind = individual_table(man)
    data, sources = ({}, [rel(MANIFEST), rel(SPLITS)])
    for spec in PROTOCOLS:
        preds, img = load_protocol_predictions(spec, seeds=pred_seeds)
        csv_path = require(METRICS / spec['csv'], f"{spec['key']} tabular baselines")
        split_ind = split_individuals(splits, man, spec['split'])
        ref = reference_frame(ind, split_ind, seed=args.seed, unit=spec['unit'])
        unit_col = 'held_out_dbs' if spec['unit'] == 'fold' else 'db'
        units = ref[['individual_id', unit_col]].drop_duplicates().rename(columns={unit_col: '_unit'})
        one_seed = preds[preds['seed'] == pred_seeds[0]]
        key = one_seed.merge(units, on='individual_id', how='left')['_unit'].astype(str)
        n_images = key.value_counts().to_dict()
        data[spec['key']] = {'spec': spec, 'img': img, 'ref': ref, 'split_ind': split_ind, 'preds': preds, 'csv': pd.read_csv(csv_path), 'n_images': n_images}
        sources += [rel(PRED / f"{spec['cfg']}_f*_s{s}_biomass_test.parquet") for s in pred_seeds]
        sources.append(rel(csv_path))
    chk = data['LODO']['img'].merge(ind[['individual_id', 'weight_g']], on='individual_id')
    wdiff = float(np.max(np.abs(chk['y_weight_g'] - chk['weight_g'])))
    tables, pooled, per_seed_rows, refit_check = ([], [], [], {})
    for k in ('LODO', 'LOSO', 'within_DB'):
        d = data[k]
        rows_k, per_seed_k = protocol_rows(d['spec'], d['img'], d['ref'], d['csv'], d['n_images'], n_boot=args.n_boot, seed=args.seed)
        tables.append(rows_k)
        per_seed_rows += per_seed_k
        pooled.append(pooled_line(d['spec'], d['img'], d['ref']))
        if d['spec']['unit'] == 'fold':
            diffs = []
            for unit, g in d['ref'].groupby('held_out_dbs'):
                y = g['y_weight_g'].values
                for col, model in (('rf_pred', 'random_forest'), ('allometric_pred', 'allometric')):
                    mine = biomass_metrics(y, g[col].values, unit='g', mape_floor=MAPE_FLOOR_G)['mae']
                    theirs, _ = _csv_reference(d['csv'], str(unit), model)
                    if np.isfinite(mine) and np.isfinite(theirs):
                        diffs.append(abs(mine - theirs))
            refit_check[k] = round(float(max(diffs)), 12) if diffs else None
    table = pd.concat(tables, ignore_index=True)

    def _paired(a: str, b: str) -> dict:
        fa = _stats_frame(data[a]['img'])
        fb = _stats_frame(data[b]['img'])
        arr = cross_seed_arrays(fa, fb, 'biomass', pred_seeds)
        r = pooled_seed_bootstrap(arr['y'], arr['preds_a'], arr['preds_b'], arr['groups'], _mae, n_boot=args.n_boot, seed=args.seed)
        per_seed = [{'seed': int(s), 'mae_a': round(_mae(arr['y'], pa), 4), 'mae_b': round(_mae(arr['y'], pb), 4), 'diff': round(_mae(arr['y'], pa) - _mae(arr['y'], pb), 4)} for s, pa, pb in zip(pred_seeds, arr['preds_a'], arr['preds_b'])]
        return {'a': data[a]['spec']['cfg'], 'b': data[b]['spec']['cfg'], 'n_paired_individuals': int(arr['n_shared']), 'n_seeds': len(pred_seeds), 'mae_a': round(float(np.mean([r['mae_a'] for r in per_seed])), 4), 'mae_b': round(float(np.mean([r['mae_b'] for r in per_seed])), 4), 'diff': round(r['delta'], 4), 'lo': round(r['lo'], 4), 'hi': round(r['hi'], 4), 'per_seed': per_seed}
    paired = {'within_db_vs_lodo': _paired('within_DB', 'LODO'), 'loso_vs_lodo': _paired('LOSO', 'LODO')}
    loso = table[table['protocol'] == 'LOSO']
    worst = loso.loc[loso['rf_length_mae_g'].idxmax()]
    flat = loso.loc[loso['test_weight_sd_g'].idxmin()]
    lodo = table[table['protocol'] == 'LODO']
    wdb = table[table['protocol'] == 'within_DB']
    allo_sub = {k: allometric_on_forest_subset(ind, data[k]['split_ind']) for k in ('LODO', 'LOSO')}
    flat_dbs = str(flat['held_out_unit']).split(';')
    l_ref = float(ind[ind['pond_or_db_id'].astype(str).isin(flat_dbs)]['length_cm'].mean())
    law_at_l_ref = {}
    for db, d in ind.groupby('pond_or_db_id'):
        d = d.dropna(subset=['weight_g', 'length_cm'])
        d = d[(d['weight_g'] > 0) & (d['length_cm'] > 0)]
        b, log_a = np.polyfit(np.log(d['length_cm'].values), np.log(d['weight_g'].values), 1)
        law_at_l_ref[str(db)] = {'b': round(float(b), 4), 'a': round(float(np.exp(log_a)), 6), 'weight_at_l_ref_g': round(float(np.exp(log_a + b * np.log(l_ref))), 3)}
    w_here = [v['weight_at_l_ref_g'] for k, v in law_at_l_ref.items() if k in flat_dbs]
    w_other = [v['weight_at_l_ref_g'] for k, v in law_at_l_ref.items() if k not in flat_dbs]
    wdb_csv = data['within_DB']['csv']
    wdb_csv_r2 = float(wdb_csv[wdb_csv['model'] == 'random_forest']['weight_r2'].mean())
    wdb_csv_mae = float(wdb_csv[wdb_csv['model'] == 'random_forest']['weight_mae'].mean())
    sd = sd_collapse(data['LODO']['img'])
    meta = {'what': 'Individual biomass metrics by protocol and held-out unit', 'n_boot': args.n_boot, 'alpha': 0.05, 'seeds': [int(s) for s in pred_seeds], 'bootstrap_and_forest_seed': int(args.seed), 'evaluation_unit': 'animal median after exponentiating view log weights', 'definitions': {'image_mae_g': 'mean of per-seed individual MAE in grams', 'image_mae_seed_sd_g': 'sample standard deviation of per-seed MAE', 'image_r2_centred': '1 - SS_res / sum_db sum_i (y_i - ybar_db)^2', 'image_mae_ci': 'individual cluster bootstrap; sampling uncertainty conditional on trained runs', 'db_mean': 'held-out database mean oracle; uses held-out labels', 'rf_length': 'reference consumes ground-truth manual measurements'}, POOLED_META_KEY: {'rows': pooled}, 'paired_comparisons_recomputed_here': paired, 'per_database_allometric_law': {'reference_length_cm': round(l_ref, 4), 'fits': law_at_l_ref}, 'allometric_variants': {'LODO': allo_sub['LODO'].to_dict(orient='records'), 'LOSO': allo_sub['LOSO'].to_dict(orient='records'), 'means_mae_g': {'LODO_as_published': round(float(lodo['allometric_mae_g'].mean()), 4), 'LODO_on_forest_subset': round(float(allo_sub['LODO']['mae_g'].mean()), 4), 'LOSO_as_published': round(float(loso['allometric_mae_g'].mean()), 4), 'LOSO_on_forest_subset': round(float(allo_sub['LOSO']['mae_g'].mean()), 4)}}, 'rf_length_only_means_mae_g': {k: round(float(table[table['protocol'] == k]['rf_length_only_mae_g'].mean()), 4) for k in ('LODO', 'LOSO', 'within_DB')}, 'rf_length_means_mae_g': {k: round(float(table[table['protocol'] == k]['rf_length_mae_g'].mean()), 4) for k in ('LODO', 'LOSO', 'within_DB')}, 'rf_refit_max_abs_mae_difference_vs_csv': refit_check, 'manifest_vs_prediction_weight_max_abs_difference_g': round(wdiff, 12), 'sd_collapse_lodo': sd.round(6).to_dict(orient='records'), 'per_seed': per_seed_rows, 'sources': sorted(set(sources))}
    write_table('t_biomass_protocols', table, meta)
    lodo_d = data['LODO']
    tier_per_seed = []
    for s_pred in pred_seeds:
        one = lodo_d['preds'][lodo_d['preds']['seed'] == s_pred]
        per_fold = {int(f): g for f, g in one.groupby('fold')}
        oof = out_of_fold_features(per_fold)
        check_no_feature_leak(lodo_d['split_ind'], oof)
        rows_s = tabular_tier_rows(ind, lodo_d['split_ind'], oof, seed=args.seed)
        rows_s.insert(0, 'seed', int(s_pred))
        tier_per_seed.append(rows_s)
    tier_all = pd.concat(tier_per_seed, ignore_index=True)
    tier = tier_all.groupby('held_out_db', as_index=False).agg({c: 'mean' for c in TIER_COLUMNS if c != 'held_out_db'})[TIER_COLUMNS]
    for col in ('n_test', 'n_train'):
        tier[col] = tier[col].astype(int)
    write_table('t_tabular_tier', tier, {'what': 'Length-only forest versus length plus out-of-fold image predictions', 'features': {'rf_length': ['length_cm'], 'rf_length_plus_image': ['length_cm', 'pred_log_weight', 'pred_length_cm']}, 'out_of_fold': 'each animal feature comes from a fold that held that animal out', 'model': {'n_estimators': 300, 'min_samples_leaf': 2, 'random_state': args.seed}, 'delta_mae': 'mae_rf_length_plus_image - mae_rf_length', 'seeds': [int(x) for x in pred_seeds], 'bootstrap_and_forest_seed': int(args.seed), 'per_seed': tier_all.to_dict(orient='records'), 'sources': [rel(MANIFEST), rel(SPLITS)] + [rel(PRED / f'E4_biomass_single_image_f*_s{x}_biomass_test.parquet') for x in pred_seeds]})
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
