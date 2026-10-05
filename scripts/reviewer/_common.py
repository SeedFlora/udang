"""Shared input loaders and atomic numerical table exports."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
OUT_TAB = REPO / 'outputs' / 'reviewer'
OUTPUTS = REPO / 'outputs'
METRICS = OUTPUTS / 'metrics'
PRED = OUTPUTS / 'predictions'
MANIFEST = REPO / 'data' / 'manifests' / 'unified_manifest.parquet'
SPLITS = REPO / 'data' / 'splits' / 'splits.parquet'

def _atomic_write(path: Path, write_fn) -> None:
    path = Path(path)
    tmp_path = path.parent / f'.{path.name}.tmp'
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

def git_head() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=REPO, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'

def rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.name

def require(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise SystemExit(f'missing input for {what}: {rel(path)}')
    return Path(path)

def write_table(name: str, df: pd.DataFrame, meta: dict) -> tuple[Path, Path]:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    csv = OUT_TAB / f'{name}.csv'
    js = OUT_TAB / f'{name}.json'
    payload = {'_meta': {**meta, 'git_head': git_head(), 'written_by': f'scripts/reviewer ({name})'}, 'rows': json.loads(df.to_json(orient='records'))}
    _atomic_write(csv, lambda tmp: df.to_csv(tmp, index=False))
    _atomic_write(js, lambda tmp: tmp.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8'))
    print(f'wrote {rel(csv)} and {rel(js)} ({len(df)} rows)')
    return (csv, js)
import sys as _sys
_sys.path.insert(0, str(REPO))

def load_preds(cfg: str, seeds=(42,), kind: str='test') -> pd.DataFrame:
    frames = []
    for s in seeds:
        for p in sorted(PRED.glob(f'{cfg}_f*_s{s}_{kind}.parquet')):
            df = pd.read_parquet(p)
            df['seed'] = s
            frames.append(df)
    if not frames:
        raise SystemExit(f'no predictions for {cfg} seeds={seeds} kind={kind}')
    return pd.concat(frames, ignore_index=True)

def attach_groups(df: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    g = man.set_index('record_id')['group_id']
    out = df.copy()
    out['group_id'] = out['record_id'].map(g).fillna(out['record_id'])
    return out
