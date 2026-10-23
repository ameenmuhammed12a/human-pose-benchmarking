"""Pull a diverse subset of JRDB-Pose per-scene COCO label files out of the remote labels zip."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import open_remote_zip

URL = "https://jrdb.erc.monash.edu/static/downloads/JRDB2022/train_dataset_with_activity/train_labels.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "jrdb", "labels", "labels_2d_pose_coco")

# One camera per scene, spread over indoor / outdoor / crowded scenes.
WANTED = [
    "bytes-cafe-2019-02-07_0_image2.json",
    "huang-2-2019-01-25_0_image2.json",
    "packard-poster-session-2019-03-20_2_image2.json",
    "tressider-2019-03-16_0_image2.json",
    "gates-basement-elevators-2019-01-17_1_image2.json",
    "memorial-court-2019-03-16_0_image2.json",
    "meyer-green-2019-03-16_0_image2.json",
    "svl-meeting-gates-2-2019-04-08_0_image4.json",
    "hewlett-packard-intersection-2019-01-24_0_image2.json",
    "stlc-111-2019-04-19_0_image2.json",
]

def main():
    os.makedirs(OUT, exist_ok=True)
    z = open_remote_zip(URL)
    by_base = {i.filename.split("/")[-1]: i for i in z.infolist()
               if "labels_2d_pose_coco" in i.filename and not i.filename.startswith("__MACOSX")}
    got = 0
    for name in WANTED:
        if name not in by_base:
            print(f"  MISSING in archive: {name}")
            continue
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  cached  {name}")
            got += 1
            continue
        data = z.read(by_base[name].filename)
        with open(dst, "wb") as f:
            f.write(data)
        print(f"  fetched {name}  ({len(data)/1e6:.2f} MB)")
        got += 1
    print(f"done: {got}/{len(WANTED)} label files in {os.path.abspath(OUT)}")

if __name__ == "__main__":
    main()
