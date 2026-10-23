# =============================================================================
#  5.2  Summary table
# =============================================================================
from IPython.display import display

if results_df.empty:
    print("No results to summarise.")
else:
    display_cols = ["dataset", "model", "images_evaluated", "gt_persons", "matched_persons",
                    "detection_recall_%", "OKS_mean(matched)", "PCK@0.5_%", "PCK@0.2_%",
                    "missing_kpt_rate_%", "mean_ms_per_image", "FPS"]
    summary_table = (results_df[display_cols]
                     .sort_values(["dataset", "model"])
                     .reset_index(drop=True)
                     .round({"detection_recall_%": 1, "OKS_mean(matched)": 3,
                             "PCK@0.5_%": 1, "PCK@0.2_%": 1, "missing_kpt_rate_%": 1,
                             "mean_ms_per_image": 1, "FPS": 2}))
    print("=" * 100)
    print("BENCHMARK SUMMARY — all metrics computed on the common 12-joint set only")
    print("=" * 100)
    display(summary_table)

    # Full table, including the diagnostic columns, goes to CSV.
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved full results table -> {csv_path}")
    if not records_df.empty:
        rec_path = RESULTS_DIR / "per_keypoint_records.csv"
        records_df.to_csv(rec_path, index=False)
        print(f"Saved {len(records_df):,} per-keypoint records -> {rec_path}")

    if SKIPPED:
        print(f"\nNot in the table (unavailable, skipped without error): "
              f"{', '.join(m.name for m in SKIPPED)}")
    print("\nReminder when reading the table: the two recall columns do NOT mean the same thing.")
    print("  YOLO      finds people itself, so its recall is genuine detection performance.")
    print("  MediaPipe is handed ground-truth person boxes (it has no multi-person detector),")
    print("            so it is never penalised for mis-localising a person; its recall counts")
    print("            only the crops in which BlazePose failed to return any pose at all.")
