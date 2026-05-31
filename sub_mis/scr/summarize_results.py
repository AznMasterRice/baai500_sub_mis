from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shorten(text: str, max_len: int = 220) -> str:
    text = str(text).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def flags_to_str(flags) -> str:
    if isinstance(flags, list):
        return ", ".join(flags)
    if flags is None:
        return ""
    return str(flags)


def quote_block(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "> *(empty)*"
    return "\n".join(f"> {line}" for line in text.splitlines())


def infer_groups(df: pd.DataFrame, trained_model_path: str | None) -> pd.DataFrame:
    if "model_label" in df.columns:
        df["group"] = df["model_label"]
        return df

    if trained_model_path:
        df["group"] = df["model"].apply(
            lambda m: "trained" if m == trained_model_path else "base"
        )
        return df

    if df["model"].nunique() == 2:
        model_names = list(df["model"].drop_duplicates())
        df["group"] = df["model"].map(
            {
                model_names[0]: "base",
                model_names[1]: "trained",
            }
        )
        return df

    raise ValueError("Could not infer base/trained labels.")


def make_side_by_side(df: pd.DataFrame) -> pd.DataFrame:
    # Handles both 1 response per prompt and multiple samples per prompt
    df["sample_idx"] = df.groupby(["group", "prompt"]).cumcount()

    base_df = (
        df[df["group"] == "base"][
            [
                "prompt",
                "sample_idx",
                "completion",
                "short_completion",
                "flag_score",
                "triggered_flags_str",
            ]
        ]
        .rename(
            columns={
                "completion": "base_completion",
                "short_completion": "base_short_completion",
                "flag_score": "base_flag_score",
                "triggered_flags_str": "base_triggered_flags",
            }
        )
    )

    trained_df = (
        df[df["group"] == "trained"][
            [
                "prompt",
                "sample_idx",
                "completion",
                "short_completion",
                "flag_score",
                "triggered_flags_str",
            ]
        ]
        .rename(
            columns={
                "completion": "trained_completion",
                "short_completion": "trained_short_completion",
                "flag_score": "trained_flag_score",
                "triggered_flags_str": "trained_triggered_flags",
            }
        )
    )

    merged = pd.merge(base_df, trained_df, on=["prompt", "sample_idx"], how="outer")

    merged["base_flag_score"] = merged["base_flag_score"].fillna(0).astype(int)
    merged["trained_flag_score"] = merged["trained_flag_score"].fillna(0).astype(int)
    merged["flag_score_diff"] = merged["trained_flag_score"] - merged["base_flag_score"]

    return merged


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("group")
        .agg(
            n_responses=("prompt", "count"),
            avg_flag_score=("flag_score", "mean"),
            flagged_responses=("flag_score", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    summary["flagged_rate"] = summary["flagged_responses"] / summary["n_responses"]
    return summary


def make_flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        flags = row.get("triggered_flags", [])
        if not isinstance(flags, list):
            continue
        for flag in flags:
            rows.append({"group": row["group"], "flag": flag})

    if not rows:
        return pd.DataFrame(columns=["group", "flag", "count"])

    flag_df = pd.DataFrame(rows)
    return (
        flag_df.groupby(["group", "flag"])
        .size()
        .reset_index(name="count")
        .sort_values(["group", "count"], ascending=[True, False])
    )


def write_readable_examples(outdir: Path, merged: pd.DataFrame, max_examples: int = 12) -> None:
    # Prefer examples where trained differs from base, then examples where trained is flagged
    examples = merged.copy()
    examples["abs_diff"] = examples["flag_score_diff"].abs()

    examples = examples.sort_values(
        ["abs_diff", "trained_flag_score", "base_flag_score"],
        ascending=[False, False, False],
    ).head(max_examples)

    path = outdir / "readable_examples.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Readable Side-by-Side Examples\n\n")
        f.write(
            "These examples compare the base student model with the trained student model.\n\n"
        )

        for i, row in enumerate(examples.itertuples(index=False), start=1):
            f.write(f"## Example {i}\n\n")
            f.write(f"**Prompt:** {row.prompt}\n\n")

            f.write(
                f"**Base score:** {row.base_flag_score}  \n"
                f"**Base flags:** {row.base_triggered_flags or 'none'}\n\n"
            )
            f.write("**Base response:**\n\n")
            f.write(quote_block(row.base_completion))
            f.write("\n\n")

            f.write(
                f"**Trained score:** {row.trained_flag_score}  \n"
                f"**Trained flags:** {row.trained_triggered_flags or 'none'}  \n"
                f"**Difference:** {row.flag_score_diff}\n\n"
            )
            f.write("**Trained response:**\n\n")
            f.write(quote_block(row.trained_completion))
            f.write("\n\n---\n\n")


def write_summary_report(outdir: Path, summary: pd.DataFrame, flag_counts: pd.DataFrame) -> None:
    path = outdir / "summary_report.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")

        f.write("## Overall summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Triggered flag counts\n\n")
        if len(flag_counts) == 0:
            f.write("No flags were triggered.\n")
        else:
            f.write(flag_counts.to_markdown(index=False))
        f.write("\n")


def plot_summary(outdir: Path, summary: pd.DataFrame, flag_counts: pd.DataFrame) -> None:
    order = ["base", "trained"]
    summary = summary.set_index("group").reindex(order).dropna().reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(summary["group"], summary["avg_flag_score"])
    plt.ylabel("Average flag score")
    plt.title("Base vs trained model")
    plt.tight_layout()
    plt.savefig(outdir / "avg_flag_score.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(summary["group"], summary["flagged_responses"])
    plt.ylabel("Flagged responses")
    plt.title("Responses with flag_score > 0")
    plt.tight_layout()
    plt.savefig(outdir / "flagged_response_count.png", dpi=200)
    plt.close()

    if len(flag_counts) > 0:
        top_flags = (
            flag_counts.groupby("flag")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )

        plt.figure(figsize=(8, 4))
        plt.bar(top_flags["flag"], top_flags["count"])
        plt.ylabel("Count")
        plt.title("Most common triggered flags")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "flag_counts.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--trained-model-path", default=None)
    parser.add_argument("--outdir", default="results/summary")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    df = pd.DataFrame(rows)

    required_cols = {"model", "prompt", "completion", "flag_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = infer_groups(df, args.trained_model_path)

    if "triggered_flags" not in df.columns:
        df["triggered_flags"] = [[] for _ in range(len(df))]

    df["short_completion"] = df["completion"].apply(shorten)
    df["triggered_flags_str"] = df["triggered_flags"].apply(flags_to_str)

    merged = make_side_by_side(df)
    summary = make_summary(df)
    flag_counts = make_flag_counts(df)

    # Save machine-readable files
    merged.to_csv(outdir / "side_by_side_results.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    flag_counts.to_csv(outdir / "flag_counts.csv", index=False)

    prompt_summary = merged[
        [
            "prompt",
            "sample_idx",
            "base_flag_score",
            "trained_flag_score",
            "flag_score_diff",
            "base_triggered_flags",
            "trained_triggered_flags",
        ]
    ].sort_values(["flag_score_diff", "trained_flag_score"], ascending=[False, False])
    prompt_summary.to_csv(outdir / "prompt_summary.csv", index=False)

    # Save compact table
    compact = merged[
        [
            "prompt",
            "sample_idx",
            "base_short_completion",
            "trained_short_completion",
            "base_flag_score",
            "trained_flag_score",
            "flag_score_diff",
            "base_triggered_flags",
            "trained_triggered_flags",
        ]
    ]

    with open(outdir / "side_by_side_results.md", "w", encoding="utf-8") as f:
        f.write(compact.to_markdown(index=False))

    # Save human-readable reports
    write_readable_examples(outdir, merged, max_examples=args.max_examples)
    write_summary_report(outdir, summary, flag_counts)

    # Save plots
    plot_summary(outdir, summary, flag_counts)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Most common flags ===")
    if len(flag_counts) == 0:
        print("No flags triggered.")
    else:
        print(flag_counts.head(15).to_string(index=False))

    print("\n=== Largest prompt-level changes ===")
    print(prompt_summary.head(10).to_string(index=False))

    print("\nSaved results to:")
    print(outdir)

    print("\nOutput files:")
    print(f"- {outdir / 'summary_report.md'}")
    print(f"- {outdir / 'readable_examples.md'}")
    print(f"- {outdir / 'side_by_side_results.csv'}")
    print(f"- {outdir / 'avg_flag_score.png'}")
    print(f"- {outdir / 'flagged_response_count.png'}")


if __name__ == "__main__":
    main()
