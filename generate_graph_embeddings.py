"""Generate graph embeddings from a SMILES CSV using UniMol."""

from __future__ import annotations

import argparse
import json
import os
import queue
from collections import deque
import multiprocessing as mp
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from SELFormerMM.utils.embedders import GraphEmbedder, save_npy

_WORKER_EMBEDDER: GraphEmbedder | None = None


def _progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def _save_progress(
    path: Path,
    *,
    total_rows: int,
    valid_smiles: int,
    completed_batches: int,
    total_batches: int,
    completed_valid_rows: int,
) -> None:
    payload = {
        "total_rows": total_rows,
        "valid_smiles": valid_smiles,
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "completed_valid_rows": completed_valid_rows,
        "updated_at_epoch": time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_embeddings_csv(
    path: Path,
    *,
    id_column: str,
    row_indices: list[int],
    ids: list[object],
    rows: np.ndarray,
    write_header: bool,
) -> None:
    columns = [f"emb_{i}" for i in range(rows.shape[1])]
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(0, id_column, ids)
    frame.insert(0, "row_idx", row_indices)
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _count_zero_rows(arr: np.ndarray, chunk_size: int = 50000) -> int:
    zero_rows = 0
    for start in range(0, arr.shape[0], chunk_size):
        chunk = np.asarray(arr[start : start + chunk_size], dtype=np.float32)
        zero_rows += int((np.linalg.norm(chunk, axis=1) == 0).sum())
    return zero_rows


def _normalize_embeddings_inplace(
    arr: np.ndarray, chunk_size: int = 50000
) -> tuple[np.ndarray, int]:
    """Mean-center + L2 normalize non-zero rows; keep zero rows unchanged."""
    emb_dim = arr.shape[1]
    sum_vec = np.zeros((emb_dim,), dtype=np.float64)
    non_zero_count = 0

    for start in range(0, arr.shape[0], chunk_size):
        chunk = np.asarray(arr[start : start + chunk_size], dtype=np.float32)
        non_zero_mask = np.linalg.norm(chunk, axis=1) != 0
        if non_zero_mask.any():
            sum_vec += chunk[non_zero_mask].sum(axis=0, dtype=np.float64)
            non_zero_count += int(non_zero_mask.sum())

    zero_count = arr.shape[0] - non_zero_count
    if non_zero_count == 0:
        return arr, zero_count

    mean_vec = (sum_vec / non_zero_count).astype(np.float32)

    for start in range(0, arr.shape[0], chunk_size):
        stop = min(start + chunk_size, arr.shape[0])
        chunk = np.asarray(arr[start:stop], dtype=np.float32)
        non_zero_mask = np.linalg.norm(chunk, axis=1) != 0
        if not non_zero_mask.any():
            continue
        centered = chunk[non_zero_mask] - mean_vec
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk[non_zero_mask] = centered / norms
        arr[start:stop] = chunk
        if hasattr(arr, "flush"):
            arr.flush()

    return arr, zero_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate graph embeddings.")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV containing SMILES.",
    )
    parser.add_argument(
        "--output_npy",
        required=True,
        help="Path to output .npy file.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output with id column and embeddings.",
    )
    parser.add_argument(
        "--smiles_column",
        default="smiles",
        help="Name of the SMILES column in the input CSV.",
    )
    parser.add_argument(
        "--id_column",
        default=None,
        help="Optional ID column to include in CSV output.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for UniMol embedding.",
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="Use GPU if available (1) or force CPU (0).",
    )
    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Optional comma-separated GPU ids for multi-GPU processing, e.g. '0,1'.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Expected UniMol embedding dimension used for zero-filled rows.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=1,
        help="Apply mean-center + L2 normalization (1) or not (0).",
    )
    parser.add_argument(
        "--batch_timeout_seconds",
        type=int,
        default=1800,
        help="Timeout per batch in seconds before splitting or zero-filling.",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=32,
        help="Minimum batch size to keep splitting; smaller failed batches become zero vectors.",
    )
    return parser.parse_args()


def _is_valid_smiles(smiles: object) -> bool:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required for graph embedding generation.") from exc

    if not isinstance(smiles, str) or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles) is not None


def _parse_gpu_ids(gpu_ids: str | None) -> list[str]:
    if not gpu_ids:
        return []
    return [gpu_id.strip() for gpu_id in gpu_ids.split(",") if gpu_id.strip()]


def _embed_subbatch_on_gpu(
    subbatch: list[tuple[int, str]],
) -> tuple[list[int], np.ndarray]:
    if _WORKER_EMBEDDER is None:
        raise RuntimeError("Graph worker embedder was not initialized.")

    indices = [row_idx for row_idx, _ in subbatch]
    smiles = [smiles for _, smiles in subbatch]
    embeddings = _WORKER_EMBEDDER.embed_smiles(
        smiles, batch_size=len(smiles), show_progress=False
    )
    return indices, embeddings


def _init_graph_worker(gpu_id: str | None, use_gpu: bool) -> None:
    global _WORKER_EMBEDDER
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    _WORKER_EMBEDDER = GraphEmbedder(use_gpu=use_gpu)


def _graph_worker_loop(
    gpu_id: str | None,
    use_gpu: bool,
    task_queue,
    result_queue,
) -> None:
    _init_graph_worker(gpu_id, use_gpu)
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, subbatch = task
        try:
            indices, embeddings = _embed_subbatch_on_gpu(subbatch)
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "task_id": task_id,
                    "error": repr(exc),
                }
            )
        else:
            result_queue.put(
                {
                    "status": "ok",
                    "task_id": task_id,
                    "indices": indices,
                    "embeddings": embeddings,
                }
            )


def _start_graph_worker(ctx, worker_id: int, gpu_id: str | None, use_gpu: bool, result_queue):
    task_queue = ctx.Queue()
    process = ctx.Process(
        target=_graph_worker_loop,
        args=(gpu_id, use_gpu, task_queue, result_queue),
    )
    process.start()
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "task_queue": task_queue,
        "process": process,
        "current": None,
    }


def _stop_graph_worker(worker) -> None:
    try:
        if worker["process"].is_alive():
            worker["task_queue"].put(None)
            worker["process"].join(timeout=2)
    except Exception:
        pass
    if worker["process"].is_alive():
        worker["process"].terminate()
        worker["process"].join(timeout=5)


def _failed_rows_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".failed.csv")


def _append_failed_rows(
    path: Path,
    *,
    id_column: str | None,
    row_indices: list[int],
    df: pd.DataFrame,
    reason: str,
    write_header: bool,
) -> None:
    rows = {
        "row_idx": row_indices,
        "reason": [reason] * len(row_indices),
    }
    if id_column and id_column in df.columns:
        rows[id_column] = df.iloc[row_indices][id_column].tolist()
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_npy)
    progress_path = _progress_path(output_path)

    df = pd.read_csv(input_path)
    if args.smiles_column not in df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in {input_path}")

    smiles_series = df[args.smiles_column]
    valid_rows = [
        (row_idx, str(smiles))
        for row_idx, smiles in enumerate(smiles_series.tolist())
        if _is_valid_smiles(smiles)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), args.embedding_dim),
    )
    embeddings[:] = 0.0
    embeddings.flush()
    gpu_ids = _parse_gpu_ids(args.gpu_ids) if args.use_gpu else []
    worker_gpu_ids = gpu_ids if gpu_ids else [None]
    total_batches = (len(valid_rows) + args.batch_size - 1) // args.batch_size if valid_rows else 0
    completed_batches = 0
    completed_valid_rows = 0
    subbatches = [
        valid_rows[i : i + args.batch_size]
        for i in range(0, len(valid_rows), args.batch_size)
    ]
    output_csv = Path(args.output_csv) if args.output_csv else None
    csv_header_written = False
    failed_rows_csv = _failed_rows_path(output_path)
    failed_rows_header_written = False

    if output_csv is not None:
        if not args.id_column or args.id_column not in df.columns:
            raise ValueError("Provide --id_column present in input CSV for CSV output.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if output_csv.exists():
            output_csv.unlink()
    if failed_rows_csv.exists():
        failed_rows_csv.unlink()

    _save_progress(
        progress_path,
        total_rows=len(df),
        valid_smiles=len(valid_rows),
        completed_batches=completed_batches,
        total_batches=total_batches,
        completed_valid_rows=completed_valid_rows,
    )

    if valid_rows:
        valid_row_mask = np.zeros(len(df), dtype=bool)
        valid_row_mask[[row_idx for row_idx, _ in valid_rows]] = True
        invalid_row_indices = np.flatnonzero(~valid_row_mask).tolist()

        if output_csv is not None and invalid_row_indices:
            _append_embeddings_csv(
                output_csv,
                id_column=args.id_column,
                row_indices=invalid_row_indices,
                ids=df.iloc[invalid_row_indices][args.id_column].tolist(),
                rows=np.zeros((len(invalid_row_indices), args.embedding_dim), dtype=np.float32),
                write_header=not csv_header_written,
            )
            csv_header_written = True

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        workers = [
            _start_graph_worker(
                ctx=ctx,
                worker_id=worker_id,
                gpu_id=gpu_id,
                use_gpu=bool(args.use_gpu),
                result_queue=result_queue,
            )
            for worker_id, gpu_id in enumerate(worker_gpu_ids)
        ]
        pending_batches = deque((batch_id, subbatch) for batch_id, subbatch in enumerate(subbatches))
        task_to_worker: dict[int, int] = {}
        next_task_id = len(subbatches)

        def assign_batch(worker) -> None:
            if worker["current"] is not None or not pending_batches:
                return
            task_id, subbatch = pending_batches.popleft()
            worker["task_queue"].put((task_id, subbatch))
            worker["current"] = {
                "task_id": task_id,
                "subbatch": subbatch,
                "started_at": time(),
            }
            task_to_worker[task_id] = worker["worker_id"]

        def finalize_terminal_failure(subbatch: list[tuple[int, str]], reason: str) -> None:
            nonlocal completed_batches, completed_valid_rows, csv_header_written, failed_rows_header_written
            indices = [row_idx for row_idx, _ in subbatch]
            if output_csv is not None:
                _append_embeddings_csv(
                    output_csv,
                    id_column=args.id_column,
                    row_indices=indices,
                    ids=df.iloc[indices][args.id_column].tolist(),
                    rows=np.zeros((len(indices), args.embedding_dim), dtype=np.float32),
                    write_header=not csv_header_written,
                )
                csv_header_written = True
            _append_failed_rows(
                failed_rows_csv,
                id_column=args.id_column,
                row_indices=indices,
                df=df,
                reason=reason,
                write_header=not failed_rows_header_written,
            )
            failed_rows_header_written = True
            completed_batches += 1
            completed_valid_rows += len(indices)
            progress.update(len(indices))
            _save_progress(
                progress_path,
                total_rows=len(df),
                valid_smiles=len(valid_rows),
                completed_batches=completed_batches,
                total_batches=total_batches,
                completed_valid_rows=completed_valid_rows,
            )

        def split_or_zero_fill(
            subbatch: list[tuple[int, str]],
            *,
            reason: str,
            worker=None,
        ) -> None:
            nonlocal total_batches, next_task_id
            batch_size = len(subbatch)
            if batch_size > args.min_batch_size:
                midpoint = max(1, batch_size // 2)
                left = subbatch[:midpoint]
                right = subbatch[midpoint:]
                right_id = next_task_id
                left_id = next_task_id + 1
                next_task_id += 2
                if right:
                    pending_batches.appendleft((right_id, right))
                if left:
                    pending_batches.appendleft((left_id, left))
                total_batches += 1
                worker_label = (
                    f"worker={worker['worker_id']} gpu={worker['gpu_id']}"
                    if worker is not None
                    else "worker=unknown"
                )
                print(
                    f"[split] {worker_label} size={batch_size} reason={reason} -> "
                    f"{len(left)} + {len(right)}"
                )
                _save_progress(
                    progress_path,
                    total_rows=len(df),
                    valid_smiles=len(valid_rows),
                    completed_batches=completed_batches,
                    total_batches=total_batches,
                    completed_valid_rows=completed_valid_rows,
                )
            else:
                print(f"[zero-fill] size={batch_size} reason={reason}")
                finalize_terminal_failure(subbatch, reason=reason)

        progress = tqdm(
            total=len(valid_rows),
            desc="Graph embeddings",
            unit="smiles",
        )
        try:
            for worker in workers:
                assign_batch(worker)

            while pending_batches or any(worker["current"] is not None for worker in workers):
                try:
                    result = result_queue.get(timeout=1)
                except queue.Empty:
                    result = None

                if result is not None:
                    task_id = result["task_id"]
                    worker_id = task_to_worker.pop(task_id, None)
                    if worker_id is not None:
                        worker = workers[worker_id]
                        current = worker["current"]
                        worker["current"] = None
                    else:
                        worker = None
                        current = None

                    if result["status"] == "ok":
                        indices = result["indices"]
                        sub_embeddings = np.asarray(result["embeddings"], dtype=np.float32)
                        if sub_embeddings.shape[1] != args.embedding_dim:
                            raise ValueError(
                                "Embedding dimension mismatch: "
                                f"expected {args.embedding_dim}, got {sub_embeddings.shape[1]}"
                            )
                        embeddings[indices, :] = sub_embeddings
                        embeddings.flush()
                        if output_csv is not None:
                            _append_embeddings_csv(
                                output_csv,
                                id_column=args.id_column,
                                row_indices=indices,
                                ids=df.iloc[indices][args.id_column].tolist(),
                                rows=sub_embeddings,
                                write_header=not csv_header_written,
                            )
                            csv_header_written = True
                        completed_batches += 1
                        completed_valid_rows += len(indices)
                        progress.update(len(indices))
                        _save_progress(
                            progress_path,
                            total_rows=len(df),
                            valid_smiles=len(valid_rows),
                            completed_batches=completed_batches,
                            total_batches=total_batches,
                            completed_valid_rows=completed_valid_rows,
                        )
                    elif current is not None:
                        split_or_zero_fill(
                            current["subbatch"],
                            reason=f"worker_error:{result['error']}",
                            worker=worker,
                        )

                    if worker is not None:
                        assign_batch(worker)

                for worker_idx, worker in enumerate(workers):
                    current = worker["current"]
                    if current is None or args.batch_timeout_seconds <= 0:
                        continue
                    elapsed = time() - current["started_at"]
                    if elapsed <= args.batch_timeout_seconds:
                        continue
                    task_id = current["task_id"]
                    task_to_worker.pop(task_id, None)
                    worker["current"] = None
                    print(
                        f"[timeout] worker={worker['worker_id']} gpu={worker['gpu_id']} "
                        f"size={len(current['subbatch'])} elapsed={int(elapsed)}s"
                    )
                    _stop_graph_worker(worker)
                    workers[worker_idx] = _start_graph_worker(
                        ctx=ctx,
                        worker_id=worker["worker_id"],
                        gpu_id=worker["gpu_id"],
                        use_gpu=bool(args.use_gpu),
                        result_queue=result_queue,
                    )
                    split_or_zero_fill(
                        current["subbatch"],
                        reason=f"timeout_after_{int(elapsed)}s",
                        worker=worker,
                    )
                    assign_batch(workers[worker_idx])
        finally:
            progress.close()
            for worker in workers:
                _stop_graph_worker(worker)

    if args.normalize:
        embeddings, zero_count = _normalize_embeddings_inplace(embeddings)
        print(
            f"Normalized graph embeddings (zero_rows={zero_count}/{embeddings.shape[0]})"
        )
    else:
        zero_count = _count_zero_rows(embeddings)

    print(f"Saved graph embeddings to: {output_path}")
    zero_rows = zero_count
    non_zero_rows = embeddings.shape[0] - zero_rows
    print(
        "Graph embedding summary: "
        f"total_rows={len(df)} valid_smiles={len(valid_rows)} "
        f"non_zero_embeddings={non_zero_rows} zero_vectors={zero_rows}"
    )

    if output_csv is not None:
        if len(valid_rows) == 0 and len(df) > 0:
            _append_embeddings_csv(
                output_csv,
                id_column=args.id_column,
                row_indices=list(range(len(df))),
                ids=df[args.id_column].tolist(),
                rows=np.asarray(embeddings, dtype=np.float32),
                write_header=not csv_header_written,
            )
        print(f"Saved graph embeddings CSV to: {output_csv}")
    if failed_rows_csv.exists():
        print(f"Saved failed graph rows to: {failed_rows_csv}")


if __name__ == "__main__":
    main()
