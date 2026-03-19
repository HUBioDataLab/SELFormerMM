"""Generate SELFIES from a SMILES CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from SELFormerMM.utils.datasets import smiles_to_selfies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SELFIES from SMILES.")
    parser.add_argument(
        "--smiles_dataset",
        required=True,
        help="Path to input CSV containing a SMILES column.",
    )
    parser.add_argument(
        "--selfies_dataset",
        required=True,
        help="Path to output CSV with a SELFIES column.",
    )
    parser.add_argument(
        "--smiles_column",
        default="smiles",
        help="Name of the SMILES column in the input CSV.",
    )
    parser.add_argument(
        "--on_error",
        default="keep",
        choices=["keep", "empty", "raise"],
        help="How to handle SMILES conversion errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.smiles_dataset)
    output_path = Path(args.selfies_dataset)

    df = pd.read_csv(input_path)
    if args.smiles_column not in df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found in {input_path}")
    if "selfies" in df.columns:
        df.drop(columns=["selfies"], inplace=True)

    tqdm.pandas(desc="Generating SELFIES")
    selfies = df[args.smiles_column].progress_apply(
        lambda smi: smiles_to_selfies(smi, on_error=args.on_error)
    )
    df.insert(0, "selfies", selfies)
    df.to_csv(output_path, index=False)
    print(f"SELFIES representation file is ready: {output_path}")


if __name__ == "__main__":
    main()
