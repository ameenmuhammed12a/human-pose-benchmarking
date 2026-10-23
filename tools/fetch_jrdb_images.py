"""Fetch only the JRDB frames we actually benchmark, out of the 21 GB remote image zip."""
import os, sys, json, glob, random
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import open_remote_zip

URL = "https://jrdb.erc.monash.edu/static/downloads/train_images.zip"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LBL_DIR = os.path.join(ROOT, "data", "jrdb", "labels", "labels_2d_pose_coco")
IMG_DIR = os.path.join(ROOT, "data", "jrdb", "images")
N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 400   # target subset size
SEED = 42

def pick_frames(n_images=None):
    """Choose frames spread evenly across scenes, preferring frames with several annotated people."""
    rng = random.Random(SEED)
    per_scene, files = {}, sorted(glob.glob(os.path.join(LBL_DIR, "*.json")))
    quota = max(1, (n_images or N_IMAGES) // max(1, len(files)))
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        n_ann = {}
        for a in d["annotations"]:
            n_ann[a["image_id"]] = n_ann.get(a["image_id"], 0) + 1
        imgs = [im for im in d["images"] if n_ann.get(im["id"], 0) > 0]
        # bias toward busier frames, but keep temporal spread by sampling from the top half
        imgs.sort(key=lambda im: -n_ann[im["id"]])
        pool = imgs[: max(quota * 4, quota)]
        per_scene[os.path.basename(fp)] = [im["file_name"] for im in rng.sample(pool, min(quota, len(pool)))]
    return per_scene

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    picks = pick_frames()
    wanted = sorted({fn for v in picks.values() for fn in v})
    print(f"{len(wanted)} frames selected across {len(picks)} scene/camera label files")

    todo = [fn for fn in wanted if not os.path.exists(os.path.join(IMG_DIR, fn.replace('/', os.sep)))]
    if not todo:
        print("all frames already cached")
        return
    # 256 KB blocks: JPEGs are ~90 KB, so a big block would waste bandwidth
    z = open_remote_zip(URL, block=256 << 10)
    by_name = {i.filename: i for i in z.infolist()}
    # fetching in archive order keeps the block cache warm (frames of a scene are contiguous)
    resolved = [(by_name["images/" + fn].header_offset, fn) for fn in todo if "images/" + fn in by_name]
    missing = [fn for fn in todo if "images/" + fn not in by_name]
    if missing:
        print(f"  {len(missing)} frames not present in archive (skipped)")
    resolved.sort()
    for k, (_, fn) in enumerate(resolved, 1):
        dst = os.path.join(IMG_DIR, fn.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            data = z.read("images/" + fn)
        except Exception as e:
            print(f"  FAILED {fn}: {e}")
            continue
        with open(dst, "wb") as f:
            f.write(data)
        if k % 20 == 0 or k == len(resolved):
            print(f"  {k}/{len(resolved)} fetched")
    print("done ->", IMG_DIR)

if __name__ == "__main__":
    main()
