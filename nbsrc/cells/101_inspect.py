def describe_json(path: Path, title: str, n_example: int = 1) -> Optional[dict]:
    """Print the real structure of a COCO-style annotation file, defensively."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not path.is_file():
        print(f"  !! FILE NOT FOUND: {path}")
        print("     Check the paths in the configuration cell (Section 0.1).")
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:                       # corrupt / truncated download
        print(f"  !! Could not parse {path.name}: {type(exc).__name__}: {exc}")
        return None

    print(f"  file            : {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
    print(f"  top-level keys  : {list(data.keys())}")
    for key, val in data.items():
        if isinstance(val, list):
            print(f"    - {key:<14}: list of {len(val):,}")
        elif isinstance(val, dict):
            print(f"    - {key:<14}: dict {list(val.keys())[:6]}")

    cats = data.get("categories") or []
    if cats:
        print("\n  categories[0]:")
        for k, v in cats[0].items():
            if k == "keypoints":
                print(f"    keypoints ({len(v)}):")
                for i, name in enumerate(v):
                    print(f"        {i:>2}  {name}")
            elif k == "skeleton":
                print(f"    skeleton  : {len(v)} limb links, e.g. {v[:4]}")
            else:
                print(f"    {k:<10}: {v}")

    if data.get("images"):
        print("\n  images[0]:")
        print("    " + json.dumps(data["images"][0], indent=2).replace("\n", "\n    "))

    anns = data.get("annotations") or []
    if anns:
        ann = anns[n_example - 1]
        print(f"\n  annotations[{n_example-1}]  (keys: {list(ann.keys())}):")
        shown = {k: v for k, v in ann.items() if k != "keypoints"}
        print("    " + json.dumps(shown, indent=2).replace("\n", "\n    "))
        kp = ann.get("keypoints")
        if kp:
            triplets = [kp[i:i + 3] for i in range(0, len(kp), 3)]
            names = cats[0]["keypoints"] if cats and "keypoints" in cats[0] else \
                    [f"kpt{i}" for i in range(len(triplets))]
            print(f"    keypoints: {len(kp)} numbers = {len(triplets)} (x, y, v) triplets")
            for name, (x, y, v) in zip(names, triplets):
                print(f"        {name:<16} x={x:<8.1f} y={y:<8.1f} v={v}")
    print()
    return data


# --- CrowdPose -------------------------------------------------------------------
_cp_raw = describe_json(CROWDPOSE_ANN_FILE, "RAW STRUCTURE 1/2 — CrowdPose annotation file")

# --- JRDB-Pose: one scene file (they all share the same schema) -------------------
_jrdb_files = sorted(JRDB_LABEL_DIR.glob("*.json")) if JRDB_LABEL_DIR.is_dir() else []
print(f"JRDB-Pose label files found: {len(_jrdb_files)}")
for f in _jrdb_files:
    print(f"   - {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
print()
_jrdb_raw = describe_json(_jrdb_files[0], "RAW STRUCTURE 2/2 — JRDB-Pose scene label file") \
            if _jrdb_files else None
