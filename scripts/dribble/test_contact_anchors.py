"""Check the contact anchors of the two clean-v0 references (24 anchors, all right foot, known frames)."""
import importlib.util
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("contact_anchors", os.path.join(here, "contact_anchors.py"))
m = importlib.util.module_from_spec(spec); sys.modules["contact_anchors"] = m; spec.loader.exec_module(m)
root = os.path.join(here, "..", "..")
EXPECT = {
    "motions/master-aligned/master_001675_001884_unitree_g1.npz": ([35, 47, 67, 85, 107, 123, 145, 163, 185, 202, 225, 240, 263, 278, 303, 325], [1, 1, 0, 0] * 4),
    "motions/master-annotated-003178-grounded/master_003178_003326_unitree_g1.npz": ([24, 52, 81, 110, 141, 172, 202, 232], [0, 1] * 4),
}
total, ok = 0, True
for path, (frames, surf) in EXPECT.items():
    a = m.load_contact_anchors(os.path.join(root, path))
    total += len(a)
    good = [x.frame for x in a] == frames and all(x.foot == "right" and x.source == "dribble_cg" for x in a) and [x.surface for x in a] == surf
    ok &= good
    withflow = sum(x.direction_local is not None for x in a)
    print(f"[{'PASS' if good else 'FAIL'}] {os.path.basename(path)}: {len(a)} anchors, frames/foot/surface match, flow fields on {withflow}/{len(a)}")
    print("   first:", a[0])
print(f"total anchors {total} (expect 24): {'PASS' if total == 24 else 'FAIL'}")
raise SystemExit(0 if ok and total == 24 else 1)
