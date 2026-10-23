"""Fetch runs of CONSECUTIVE annotated JRDB frames, for the temporal-stability analysis.

tools/fetch_jrdb_images.py samples frames scattered across each scene, which is right for a
static accuracy benchmark but useless for measuring frame-to-frame jitter. This tool instead
picks the longest unbroken run of annotated frames in each of the busiest scenes and fetches
that run intact, so consecutive predictions can be compared directly.
"""
import os, sys, json, glob, re, time
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import open_remote_zip, HttpRangeFile, read_member_at

URL = "https://jrdb.erc.monash.edu/static/downloads/train_images.zip"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LBL_DIR = os.path.join(ROOT, "data", "jrdb", "labels", "labels_2d_pose_coco")
SEQ_DIR = os.path.join(ROOT, "data", "jrdb", "sequences")
# The archive index is 22.5 MB and this server serves ~20 KB/s, so reading it costs about
# half an hour. The header offsets we resolve from it are cached here, which makes every
# later run — a resumed download, or a different choice of sequence — effectively instant.
OFFSET_CACHE = os.path.join(ROOT, "data", "jrdb", "archive_offsets.json")

# Two different block sizes, because the two phases want opposite things. Reading the index
# is one long contiguous run, so larger blocks mean fewer round trips. Fetching frames is the
# opposite: a block much larger than a ~90 KB JPEG is wasted bandwidth for scattered frames.
# 1 MB is the ceiling: this server sustains roughly 30 KB/s, and requests much above a
# megabyte were observed to stall rather than merely run slowly.
INDEX_BLOCK = 1 << 20
# The fetch block is small on purpose. Consecutive frames of one camera turn out to sit
# 3-6 MB apart in this archive (it is ordered by frame, not by camera), so blocks never get
# reused across frames -- every megabyte fetched beyond the ~90 KB JPEG itself is wasted.
FETCH_BLOCK = 64 << 10

N_SCENES = 4            # scenes contributing a sequence
SEQ_LEN = 60            # consecutive frames taken from each


def annotated_runs(label_path):
    """Return (frame_numbers, file_name_by_frame, mean_people) for one scene label file."""
    d = json.load(open(label_path, encoding="utf-8"))
    by_id = {im["id"]: im for im in d["images"]}
    count = {}
    for a in d["annotations"]:
        count[a["image_id"]] = count.get(a["image_id"], 0) + 1
    if not count:
        return [], {}, 0.0
    frames, name_by_frame = [], {}
    for img_id in count:
        fn = by_id[img_id]["file_name"]
        n = int(re.search(r"(\d+)\.jpg", fn).group(1))
        frames.append(n)
        name_by_frame[n] = fn
    frames.sort()
    mean_people = sum(count.values()) / len(count)
    return frames, name_by_frame, mean_people


def longest_run(frames):
    """Longest stretch of frames with no gap."""
    if not frames:
        return []
    best, cur = [frames[0]], [frames[0]]
    for prev, cur_f in zip(frames, frames[1:]):
        if cur_f == prev + 1:
            cur.append(cur_f)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [cur_f]
    return cur if len(cur) > len(best) else best


def pick_sequences():
    """Choose the busiest scenes that offer a long enough unbroken run."""
    options = []
    for fp in sorted(glob.glob(os.path.join(LBL_DIR, "*.json"))):
        frames, names, density = annotated_runs(fp)
        run = longest_run(frames)
        if len(run) < SEQ_LEN:
            continue
        # take the middle of the run: scene starts are often sparsely populated
        mid = run[len(run) // 2 - SEQ_LEN // 2: len(run) // 2 + (SEQ_LEN + 1) // 2]
        options.append((density, os.path.basename(fp), [names[f] for f in mid]))
    options.sort(key=lambda o: -o[0])         # busiest scenes make the sternest test
    return options[:N_SCENES]


def main():
    picks = pick_sequences()
    if not picks:
        print("no scene has a long enough annotated run — nothing to fetch")
        return
    print(f"{len(picks)} sequences x {SEQ_LEN} consecutive frames:")
    wanted = []
    for density, scene, files in picks:
        print(f"  {scene[:44]:46s} {files[0].split('/')[-1]} .. {files[-1].split('/')[-1]}"
              f"   ({density:.1f} people/frame)")
        wanted += files

    todo = [fn for fn in wanted
            if not os.path.exists(os.path.join(SEQ_DIR, fn.replace("/", os.sep)))]
    if not todo:
        print("all sequence frames already cached")
        return
    print(f"\nfetching {len(todo)} frames over HTTP range requests...")

    cache = {}
    if os.path.isfile(OFFSET_CACHE):
        try:
            cache = json.load(open(OFFSET_CACHE, encoding="utf-8"))
        except Exception:
            cache = {}

    z = None
    if all(fn in cache for fn in todo):
        # Cached offsets let us skip the central directory entirely and read each frame
        # straight out of its local file header.
        print("  (header offsets served from the local index cache — no index read needed)")
        fp = HttpRangeFile(URL, block=FETCH_BLOCK)
        resolved = sorted((cache[fn], fn) for fn in todo)
        missing = []
    else:
        print("  reading the 22.5 MB archive index - about 12 min at this server's")
        print("  speed, and cached afterwards so it happens only once.")
        t0 = time.time()
        z = open_remote_zip(URL, block=INDEX_BLOCK)
        by_name = {i.filename: i for i in z.infolist()}
        print(f"  index read in {(time.time()-t0)/60:.0f} min — {len(by_name):,} entries")
        for fn in wanted:
            if "images/" + fn in by_name:
                cache[fn] = by_name["images/" + fn].header_offset
        os.makedirs(os.path.dirname(OFFSET_CACHE), exist_ok=True)
        with open(OFFSET_CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        print(f"  offsets cached -> {OFFSET_CACHE}")
        resolved = sorted((cache[fn], fn) for fn in todo if fn in cache)
        missing = [fn for fn in todo if fn not in cache]
        # the index read filled the block cache with 4 MB blocks that are useless for
        # fetching individual frames, so start the fetch phase from a clean, smaller cache
        fp = z.fp
        fp.block = FETCH_BLOCK
        fp._cache.clear()
    if missing:
        print(f"  {len(missing)} frames not present in archive (skipped)")
    t_fetch = time.time()
    for k, (_, fn) in enumerate(resolved, 1):
        dst = os.path.join(SEQ_DIR, fn.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            data = (read_member_at(fp, cache[fn]) if z is None
                    else z.read("images/" + fn))
        except Exception as exc:
            print(f"  FAILED {fn}: {exc}")
            continue
        with open(dst, "wb") as f:
            f.write(data)
        if k % 10 == 0 or k == len(resolved):
            rate = k / max(1e-9, time.time() - t_fetch)
            print(f"  {k}/{len(resolved)} fetched  ({rate*60:.0f}/min, "
                  f"ETA {(len(resolved)-k)/max(1e-9, rate)/60:.0f} min)", flush=True)
    print("done ->", SEQ_DIR)


if __name__ == "__main__":
    main()
