"""CPU regression checks; no Kaggle credentials or competition data required.

Run: python Eugene/test_graph_pipeline.py
Uses a synthetic compressed graph to check the notebook's actual definitions.
The loading benchmark is illustrative, not a full-training speedup claim.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


def load_definitions(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    ns = {}
    for i in (4, 8, 10, 12, 14, 16):
        tree = ast.parse("".join(cells[i]["source"]))
        if i in (10, 16):
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(notebook), "exec"), ns)
    return ns


def main():
    notebook = Path(__file__).resolve().with_name("graph.ipynb")
    original_dir = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            ns = load_definitions(notebook)
            ns["GNN_EPOCHS"] = 2
            ns["GNN_MAX_TRAIN_CONFIGS_PER_FILE"] = 32
            ns["GNN_MAX_VALID_CONFIGS_PER_FILE"] = 32
            ns["GNN_HIDDEN_DIM"] = 32
            random = np.random.default_rng(81)
            config = random.integers(-1, 6, (2048, 64, 18), dtype=np.int8)
            data = dict(
                node_feat=random.uniform(0, 100, (96, 8)).astype(np.float32),
                node_opcode=random.integers(0, 20, 96),
                edge_index=np.column_stack([np.arange(95), np.arange(1, 96)]),
                node_config_ids=np.arange(64), node_config_feat=config,
                config_runtime=(10 + (config[:, :, 0] + 1).sum(axis=1)).astype(np.float32),
            )
            train = Path(tmp) / "train.npz"
            valid = Path(tmp) / "valid.npz"
            np.savez_compressed(train, **data)
            np.savez_compressed(valid, **data)
            # Synthetic train/valid share inputs solely to exercise mechanics, not generalization.
            model = ns["LayoutGNNRanker"]()
            entry = model.cache.get(train, model)
            graph = model.device_graph(entry)
            batch = np.array([0, 17, 201, 700])
            old_tensor = model.config_tensor(data, batch, len(graph[0]), graph[6], graph[7])
            new_tensor = model.cached_config_tensor(entry, batch, graph)
            torch.testing.assert_close(new_tensor, old_tensor, rtol=0, atol=0)
            model.model.eval()
            base, opcode, src, dst, degree, mask, _, _ = graph
            old_scores = model.model(base, opcode, old_tensor, src, dst, degree, mask)
            new_scores = model.forward_batch(entry, batch, graph)
            torch.testing.assert_close(new_scores, old_scores, rtol=0, atol=0)
            # Gradients must also be identical at a fixed model state and batch.
            old_scores.sum().backward()
            gradients = {k: p.grad.clone() for k, p in model.model.named_parameters()}
            model.model.zero_grad(set_to_none=True)
            model.forward_batch(entry, batch, graph).sum().backward()
            for k, p in model.model.named_parameters():
                torch.testing.assert_close(p.grad, gradients[k], rtol=0, atol=0)
            assert model.cache.get(train, model) is entry
            assert model.cache.extractions == 1 and model.cache.graph_preparations == 1

            # Uniform validation selection is independent of runtime labels.
            a = ns["choose_config_indices"](data, "valid", 32)
            permuted = dict(data, config_runtime=data["config_runtime"][::-1])
            b = ns["choose_config_indices"](permuted, "valid", 32)
            np.testing.assert_array_equal(a, b)
            assert ns["sampled_kendall_score"](np.arange(10), np.zeros(10)) == 0
            assert np.isfinite(ns["choose_runtime_stratified_indices"](np.array([np.nan, -1, 10]), 4)).all()
            np.testing.assert_array_equal(ns["choose_runtime_stratified_indices"](np.array([np.nan, -1, 10]), 4), [2])

            # A fresh RAM cache reuses extracted disk arrays, even with no LRU capacity.
            empty = ns["PreparedGraphCache"](max_mb=0)
            empty.get(train, model)
            empty.get(train, model)
            assert empty.extractions == 0 and empty.bytes == 0 and not empty.entries

            started = time.perf_counter()
            with np.load(train) as archive:
                for start in range(0, 128, 4):
                    model.config_tensor(archive, np.arange(start, start + 4), len(base), graph[6], graph[7])
            old_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for start in range(0, 128, 4):
                model.cached_config_tensor(entry, np.arange(start, start + 4), graph)
            cached_seconds = time.perf_counter() - started
            print(f"Synthetic 32-batch input construction: original={old_seconds:.4f}s, cached={cached_seconds:.4f}s")

            trained = ns["LayoutGNNRanker"]().fit("synthetic", [train], valid_files=[valid])
            assert len(trained.history) == 2
            assert trained.best_epoch in (1, 2)
            checkpoint = torch.load(Path(tmp) / "gnn_runs/synthetic/best.pt", weights_only=True)
            for k, v in trained.model.state_dict().items():
                torch.testing.assert_close(v.cpu(), checkpoint["model_state"][k], rtol=0, atol=0)
            indices, predictions = trained.predict_file(valid, max_configs=32)
            assert len(indices) == len(predictions) == 32 and np.isfinite(predictions).all()
            assert not trained.model.training
            assert trained.cache.graph_preparations == 2
            assert trained.cache.extractions == 1  # train was already extracted above
            assert (Path(tmp) / "gnn_runs/synthetic/history.csv").exists()
            print("PASS: cache parity, gradient parity, reuse, bounded LRU, label-independent sampling, training, best-checkpoint restoration and prediction")
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    main()
