"""Exercise the full notebook on tiny synthetic data with reduced fit iterations.

No downloads, credentials or competition data. This verifies pipeline wiring and
submission invariants, not leaderboard performance. Run from any working directory.
"""
import ast
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import warnings
import zipfile
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


def main():
    notebook = json.loads(Path(__file__).with_name('SC4000_Full_Hybrid.ipynb').read_text())
    sources = {i: ''.join(c['source']) for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code'}
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            random = np.random.default_rng(42)
            collections = ['tile:xla', 'layout:xla:default', 'layout:xla:random', 'layout:nlp:default', 'layout:nlp:random']
            rows = []
            for collection in collections:
                for split in ['train', 'valid', 'test']:
                    folder = Path('data/npz_all/npz').joinpath(*collection.split(':'), split)
                    folder.mkdir(parents=True)
                    arrays = dict(node_feat=random.uniform(0, 10, (12, 8)).astype(np.float32),
                                  node_opcode=np.arange(12),
                                  edge_index=np.column_stack([np.arange(11), np.arange(1, 12)]),
                                  config_runtime=random.uniform(1, 10, 32).astype(np.float32))
                    if collection.startswith('tile'):
                        arrays['config_feat'] = random.uniform(0, 5, (32, 6)).astype(np.float32)
                    else:
                        arrays['node_config_ids'] = np.arange(6)
                        arrays['node_config_feat'] = random.integers(-1, 4, (32, 6, 18), dtype=np.int8)
                    if split == 'test': arrays.pop('config_runtime')
                    np.savez_compressed(folder / 'bert_example.npz', **arrays)
                rows.append({'ID': f'{collection}:bert_example', 'TopConfigs': ';'.join(map(str, range(5 if collection.startswith('tile') else 32)))})
            pd.DataFrame(rows).to_csv('data/sample_submission.csv', index=False)
            ns = dict(globals(), display=lambda value: None, RANDOM_SEED=42,
                      rng=np.random.default_rng(42), XGBOOST_AVAILABLE=False,
                      LIGHTGBM_AVAILABLE=False, CPU_THREADS=1, FULL_OUTPUT=Path('full_run'))
            ns['FULL_OUTPUT'].mkdir()
            warnings.filterwarnings('ignore')
            # Skip installation/data-download cells; dependencies are imported above.
            for i in [5, 6, 8, 9, 10, 11]: exec(sources[i], ns)
            # Execute setup/definitions from baseline training before its run loops.
            body = ast.parse(sources[13]).body[0].body
            cut = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'experiment_results' for t in x.targets))
            exec(compile(ast.Module(body=body[:cut], type_ignores=[]), '<baseline-setup>', 'exec'), ns)
            original_factory = ns['make_runtime_model']
            def small_model(kind):
                model = original_factory(kind)
                if kind == 'mlp': model.set_params(mlpregressor__max_iter=2)
                else: model.set_params(max_iter=3)
                return model
            ns['make_runtime_model'] = small_model
            with threadpool_limits(limits=1):
                exec(compile(ast.Module(body=body[cut:], type_ignores=[]), '<baseline-fit>', 'exec'), ns)
            exec(sources[14], ns)
            exec(sources[16], ns)
            # Run the actual GNN orchestration, reducing epochs only for this smoke test.
            exec(sources[18].replace('gnn["GNN_EPOCHS"] = 20', 'gnn["GNN_EPOCHS"] = 2'), ns)
            exec(sources[20], ns)
            baseline = pd.read_csv('full_run/submission_baseline.csv')
            hybrid = pd.read_csv('full_run/submission_hybrid.csv')
            mask = ~baseline.ID.str.startswith(('layout:xla:default:', 'layout:xla:random:'))
            pd.testing.assert_frame_equal(baseline[mask], hybrid[mask])
            assert len(hybrid) == 5
            assert len(joblib.load('full_run/models/ensemble_members_by_collection.joblib')) == 5
            assert len(pd.read_csv('full_run/validation_comparison.csv')) == 4
            # Invalid replacements must be rejected by the export validator.
            bad = ns['hybrid_submission'].copy()
            bad.loc[0, 'TopConfigs'] = '0;0;1;2;3'
            try:
                ns['validate_candidate'](bad, ns['baseline_submission'], ns['template'], ns['config_counts'])
                raise AssertionError('Duplicate configuration was not rejected')
            except ValueError:
                pass
            print('PASS: five-collection baseline fit/save, two-GNN fit, shared comparison, CSV export, untouched rows and invalid-index rejection')
        finally:
            os.chdir(original)


if __name__ == '__main__': main()
