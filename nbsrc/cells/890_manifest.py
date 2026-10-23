# =============================================================================
#  A final consolidated printout of everything this notebook produced
# =============================================================================
print("=" * 88)
print("RUN MANIFEST")
print("=" * 88)
print(f"  finished                : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  random seed             : {RANDOM_SEED}")
print(f"  device                  : CPU only")
print(f"  datasets loaded         : "
      f"{', '.join(f'{d.name} ({len(d.samples)} images)' for d in DATASETS.values()) or 'none'}")
print(f"  models benchmarked      : {', '.join(m.name for m in MODELS) or 'none'}")
print(f"  models skipped          : {', '.join(m.name for m in SKIPPED) or 'none'}")
print(f"  common joints evaluated : {N_COMMON} ({', '.join(COMMON_KEYPOINTS)})")
if not records_df.empty:
    print(f"  keypoint records        : {len(records_df):,}")

print("\n  Files written to " + str(RESULTS_DIR) + ":")
for f in sorted(RESULTS_DIR.glob("*")):
    print(f"      {f.name:<34} {f.stat().st_size/1024:8.1f} KB")

if not results_df.empty:
    print("\n" + "=" * 88)
    print("HEADLINE NUMBERS (repeated here so the notebook ends with its own result)")
    print("=" * 88)
    for _, r in results_df.sort_values(["dataset", "model"]).iterrows():
        print(f"  {r['dataset']:<11} {r['model']:<16} "
              f"PCK@0.5 {r['PCK@0.5_%']:5.1f}%  OKS {r['OKS_mean(matched)']:.3f}  "
              f"missing {r['missing_kpt_rate_%']:5.1f}%  {r['FPS']:5.2f} FPS")

# Release the MediaPipe graph cleanly.
try:
    mp_model.close()
except Exception:
    pass
print("\nNotebook run complete.")
