"""Pretrain SELFormerMM backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from SELFormerMM.models.multimodal_roberta import MultimodalRoberta
from SELFormerMM.pretrainer import Pretrainer, SINCERELoss
from SELFormerMM.utils.datasets import PretrainDataset, MultimodalCollator
from SELFormerMM.utils.embedders import save_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain multimodal backbone.")
    parser.add_argument("--selfies_csv", required=True, help="CSV with SELFIES.")
    parser.add_argument(
        "--selfies_column", default="selfies", help="SELFIES column name."
    )
    parser.add_argument("--graph_embs", default=None, help="Path to graph .npy.")
    parser.add_argument("--text_embs", default=None, help="Path to text .npy.")
    parser.add_argument("--kg_embs", default=None, help="Path to KG .npy.")
    parser.add_argument("--model_path", required=True, help="HF model id or path.")
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer path.")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=267)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_dir", required=True, help="Directory to save model.")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--save_embeddings",
        default=None,
        help="Optional .npy path to save final embeddings.",
    )
    return parser.parse_args()


def _load_optional_npy(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    return np.load(path).astype(np.float32)


def _count_empty_rows(arr: np.ndarray) -> int:
    """Count rows that are all zeros."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={arr.shape}")
    return int((~np.any(arr != 0, axis=1)).sum())


def _safe_float(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _batch_similarity_metrics(
    logits: torch.Tensor, labels: torch.Tensor, knn_k: int = 5
) -> tuple[float | None, float | None, float | None, float | None]:
    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.detach().cpu()
    unique_labels = labels_cpu.unique()

    silhouette = None
    if unique_labels.numel() > 1 and logits_cpu.shape[0] > unique_labels.numel():
        silhouette = float(
            silhouette_score(logits_cpu.numpy(), labels_cpu.numpy())
        )

    normalized = F.normalize(logits_cpu, p=2, dim=1)
    cosine = normalized @ normalized.T
    same_label = labels_cpu.unsqueeze(1) == labels_cpu.unsqueeze(0)
    upper_triangle = torch.triu(torch.ones_like(same_label, dtype=torch.bool), diagonal=1)
    intra_mask = same_label & upper_triangle
    inter_mask = (~same_label) & upper_triangle

    intra_sim = (
        float(cosine[intra_mask].mean().item()) if intra_mask.any() else None
    )
    inter_sim = (
        float(cosine[inter_mask].mean().item()) if inter_mask.any() else None
    )

    knn_acc = None
    if logits_cpu.shape[0] > 1 and unique_labels.numel() > 1:
        k = min(knn_k, logits_cpu.shape[0])
        preds = (
            KNeighborsClassifier(k, metric="cosine")
            .fit(logits_cpu.numpy(), labels_cpu.numpy())
            .predict(logits_cpu.numpy())
        )
        knn_acc = float(accuracy_score(labels_cpu.numpy(), preds))

    return silhouette, intra_sim, inter_sim, knn_acc


def _evaluate_split(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    prefix: str,
    num_views: int = 4,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    silhouettes: list[float] = []
    intra_sims: list[float] = []
    inter_sims: list[float] = []
    knn_accs: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch["graph_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
            kg_emb = batch["kg_emb"].to(device)

            batch_size = input_ids.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=device).repeat(
                num_views
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                text_emb=text_emb,
                kg_emb=kg_emb,
            )
            loss = loss_fn(logits, labels)

            total_loss += _safe_float(loss)
            batch_count += 1

            silhouette, intra_sim, inter_sim, knn_acc = _batch_similarity_metrics(
                logits, labels
            )
            if silhouette is not None:
                silhouettes.append(silhouette)
            if intra_sim is not None:
                intra_sims.append(intra_sim)
            if inter_sim is not None:
                inter_sims.append(inter_sim)
            if knn_acc is not None:
                knn_accs.append(knn_acc)

    return {
        f"{prefix}_loss": total_loss / max(batch_count, 1),
        f"{prefix}_silhouette": float(np.mean(silhouettes)) if silhouettes else 0.0,
        f"{prefix}_intra_sim": float(np.mean(intra_sims)) if intra_sims else 0.0,
        f"{prefix}_inter_sim": float(np.mean(inter_sims)) if inter_sims else 0.0,
        f"{prefix}_knn_acc": float(np.mean(knn_accs)) if knn_accs else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print("[pretrain] starting")
    print(f"[pretrain] device={device} batch_size={args.batch_size} max_len={args.max_len} lr={args.lr} epochs={args.epochs} val_frac={args.val_frac} save_every={args.save_every}")
    print(f"[pretrain] selfies_csv={args.selfies_csv} model_path={args.model_path}")

    print("[pretrain] loading selfies csv...")
    df = pd.read_csv(args.selfies_csv)
    if args.selfies_column not in df.columns:
        raise ValueError(f"Column '{args.selfies_column}' not found.")
    selfies = df[args.selfies_column].fillna("").astype(str).tolist()
    print(f"[pretrain] loaded {len(selfies):,} selfies strings (column='{args.selfies_column}')")

    print("[pretrain] loading modality embeddings (npy)...")
    graph = _load_optional_npy(args.graph_embs)
    text = _load_optional_npy(args.text_embs)
    kg = _load_optional_npy(args.kg_embs)
    if graph is None:
        print("[pretrain] graph_embs: None (will use zeros)")
    else:
        print(f"[pretrain] graph_embs: {args.graph_embs} shape={tuple(graph.shape)}")
        print(f"[pretrain] graph_embs empty_rows={_count_empty_rows(graph):,}/{graph.shape[0]:,}")
    if text is None:
        print("[pretrain] text_embs: None (will use zeros)")
    else:
        print(f"[pretrain] text_embs: {args.text_embs} shape={tuple(text.shape)}")
        print(f"[pretrain] text_embs empty_rows={_count_empty_rows(text):,}/{text.shape[0]:,}")
    if kg is None:
        print("[pretrain] kg_embs: None (will use zeros)")
    else:
        print(f"[pretrain] kg_embs: {args.kg_embs} shape={tuple(kg.shape)}")
        print(f"[pretrain] kg_embs empty_rows={_count_empty_rows(kg):,}/{kg.shape[0]:,}")

    print("[pretrain] loading config/tokenizer/backbone...")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    hf_backbone = AutoModel.from_pretrained(args.model_path)
    print("[pretrain] initializing multimodal model and loading roberta weights...")
    model = MultimodalRoberta(config)
    model.roberta.load_state_dict(hf_backbone.state_dict(), strict=False)
    model.to(device)

    # Print model summary and parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pretrain] model class: {model.__class__.__name__}")
    print(f"[pretrain] total_params={total_params:,} trainable_params={trainable_params:,}")
    try:
        print("[pretrain] model architecture:\n" + str(model))
    except Exception:
        print("[pretrain] model architecture: <unavailable>")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for train/val split in pretraining."
        ) from exc

    indices = np.arange(len(selfies))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, random_state=42
    )
    print(f"[pretrain] split sizes: train={len(train_idx):,} val={len(val_idx):,}")

    print("[pretrain] building datasets/dataloaders...")
    train_dataset = PretrainDataset(
        selfies=[selfies[i] for i in train_idx],
        tokenizer=tokenizer,
        max_len=args.max_len,
        graph_emb=graph[train_idx] if graph is not None else None,
        text_emb=text[train_idx] if text is not None else None,
        kg_emb=kg[train_idx] if kg is not None else None,
    )
    val_dataset = PretrainDataset(
        selfies=[selfies[i] for i in val_idx],
        tokenizer=tokenizer,
        max_len=args.max_len,
        graph_emb=graph[val_idx] if graph is not None else None,
        text_emb=text[val_idx] if text is not None else None,
        kg_emb=kg[val_idx] if kg is not None else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MultimodalCollator(),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultimodalCollator(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultimodalCollator(),
    )
    print(
        f"[pretrain] train_steps_per_epoch={len(train_loader):,} "
        f"train_eval_steps={len(train_eval_loader):,} val_steps={len(val_loader):,}"
    )

    print("[pretrain] initializing optimizer/loss/trainer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = SINCERELoss()
    trainer = Pretrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    def _pairwise_metrics(embeds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        logits = embeds @ embeds.T
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        same = same & ~eye
        diff = ~same & ~eye
        pos = logits[same].mean().item() if same.any() else 0.0
        neg = logits[diff].mean().item() if diff.any() else 0.0
        return pos, neg

    print("[pretrain] training loop...")
    baseline_train_metrics = _evaluate_split(
        model=model,
        dataloader=train_eval_loader,
        loss_fn=loss_fn,
        device=device,
        prefix="train",
        num_views=trainer.num_views,
    )
    baseline_val_metrics = _evaluate_split(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        prefix="val",
        num_views=trainer.num_views,
    )
    print("[pretrain] baseline before training")
    print(
        f"[pretrain]   train: "
        f"loss={baseline_train_metrics['train_loss']:.4f} "
        f"silhouette={baseline_train_metrics['train_silhouette']:.4f} "
        f"intra_sim={baseline_train_metrics['train_intra_sim']:.4f} "
        f"inter_sim={baseline_train_metrics['train_inter_sim']:.4f} "
        f"knn_acc={baseline_train_metrics['train_knn_acc']:.4f}"
    )
    print(
        f"[pretrain]   val: "
        f"loss={baseline_val_metrics['val_loss']:.4f} "
        f"silhouette={baseline_val_metrics['val_silhouette']:.4f} "
        f"intra_sim={baseline_val_metrics['val_intra_sim']:.4f} "
        f"inter_sim={baseline_val_metrics['val_inter_sim']:.4f} "
        f"knn_acc={baseline_val_metrics['val_knn_acc']:.4f}"
    )
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        train_loop_loss = trainer.train_epoch(train_loader)
        train_metrics = _evaluate_split(
            model=model,
            dataloader=train_eval_loader,
            loss_fn=loss_fn,
            device=device,
            prefix="train",
            num_views=trainer.num_views,
        )
        val_metrics = _evaluate_split(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            prefix="val",
            num_views=trainer.num_views,
        )
        epoch_time = time.time() - epoch_t0
        print(f"[pretrain] epoch {epoch + 1}/{args.epochs} time={epoch_time:.1f}s")
        print(
            f"[pretrain]   train: "
            f"loop_loss={train_loop_loss:.4f} "
            f"eval_loss={train_metrics['train_loss']:.4f} "
            f"silhouette={train_metrics['train_silhouette']:.4f} "
            f"intra_sim={train_metrics['train_intra_sim']:.4f} "
            f"inter_sim={train_metrics['train_inter_sim']:.4f} "
            f"knn_acc={train_metrics['train_knn_acc']:.4f}"
        )
        print(
            f"[pretrain]   val: "
            f"loss={val_metrics['val_loss']:.4f} "
            f"silhouette={val_metrics['val_silhouette']:.4f} "
            f"intra_sim={val_metrics['val_intra_sim']:.4f} "
            f"inter_sim={val_metrics['val_inter_sim']:.4f} "
            f"knn_acc={val_metrics['val_knn_acc']:.4f}"
        )

        # simple validation metrics
        with torch.no_grad():
            batch = next(iter(val_loader))
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                graph_emb=batch["graph_emb"].to(device),
                text_emb=batch["text_emb"].to(device),
                kg_emb=batch["kg_emb"].to(device),
            )
            labels = trainer._build_labels(batch["input_ids"].size(0))
            pos_cos, neg_cos = _pairwise_metrics(logits, labels)
            print(f"[pretrain]   val_pos_cos={pos_cos:.4f} val_neg_cos={neg_cos:.4f}")

        if (epoch + 1) % args.save_every == 0:
            epoch_dir = Path(args.save_dir) / f"epoch_{epoch + 1:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), epoch_dir / "pytorch_model.bin")
            config.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            print(f"[pretrain] saved checkpoint: {epoch_dir}")

    print("[pretrain] saving final model...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[pretrain] saved final model: {save_dir}")

    if args.save_embeddings:
        print("[pretrain] producing embeddings for full dataset...")
        full_dataset = PretrainDataset(
            selfies=selfies,
            tokenizer=tokenizer,
            max_len=args.max_len,
            graph_emb=graph,
            text_emb=text,
            kg_emb=kg,
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=MultimodalCollator(),
        )
        model.eval()
        out_path = Path(args.save_embeddings)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total_rows = len(full_dataset)
        writer = None
        offset = 0

        with torch.no_grad():
            for step, batch in enumerate(full_loader, start=1):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    graph_emb=batch["graph_emb"].to(device),
                    text_emb=batch["text_emb"].to(device),
                    kg_emb=batch["kg_emb"].to(device),
                )
                chunk = logits.detach().cpu().numpy()
                if writer is None:
                    embed_dim = chunk.shape[1]
                    writer = np.memmap(
                        out_path,
                        mode="w+",
                        dtype=chunk.dtype,
                        shape=(total_rows, embed_dim),
                    )
                rows = chunk.shape[0]
                writer[offset : offset + rows] = chunk
                offset += rows
                if step == 1 or step % 50 == 0 or step == len(full_loader):
                    print(f"[pretrain]   embedding write: {offset:,}/{total_rows:,} rows")

        if writer is not None:
            del writer

        print(f"[pretrain] saved embeddings to: {args.save_embeddings}")

    print(f"[pretrain] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
