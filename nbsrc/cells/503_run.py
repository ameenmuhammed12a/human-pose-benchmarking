# =============================================================================
#  Run every available model over every loaded dataset
# =============================================================================
BENCH_TARGETS = []
if crowdpose.samples:
    BENCH_TARGETS.append((crowdpose, cp_samples, CROWDPOSE14_TO_COMMON))
if jrdb.samples:
    BENCH_TARGETS.append((jrdb, jr_samples, JRDB17_TO_COMMON))

summaries, all_records = [], []
if not MODELS or not BENCH_TARGETS:
    print("Nothing to benchmark (no available model, or no dataset loaded).")
else:
    total = len(MODELS) * len(BENCH_TARGETS)
    print(f"Benchmarking {len(MODELS)} model(s) on {len(BENCH_TARGETS)} dataset(s) "
          f"= {total} runs, CPU only.\n")
    run_i = 0
    bench_t0 = time.perf_counter()
    for ds, samples, gt_map in BENCH_TARGETS:
        for model in MODELS:
            run_i += 1
            print(f"[{run_i}/{total}] {model.name} on {ds.name} "
                  f"({len(samples)} images)...", flush=True)
            try:
                s, df = evaluate(model, ds, samples, gt_map)
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                continue
            summaries.append(s)
            if not df.empty:
                all_records.append(df)
            print(f"    done — PCK@0.5 {s['PCK@0.5_%']:.1f}% | OKS {s['OKS_mean(matched)']:.3f} "
                  f"| missing {s['missing_kpt_rate_%']:.1f}% "
                  f"| {s['mean_ms_per_image']:.0f} ms/img ({s['FPS']:.1f} FPS)\n", flush=True)
    print(f"All runs finished in {time.perf_counter() - bench_t0:.0f} s.")

records_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
results_df = pd.DataFrame(summaries)
print(f"\nCollected {len(records_df):,} per-keypoint evaluation records "
      f"across {len(results_df)} model x dataset runs.")
