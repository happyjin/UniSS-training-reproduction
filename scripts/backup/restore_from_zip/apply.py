"""Restore unix modes and missing symlinks under ROOT from the zip-backup manifests.
Never deletes or overwrites: chmod only on existing regular files whose mode differs; symlinks only where nothing
exists at the path (and the parent dir exists). --apply to act; default is a dry run."""
import glob, os, stat, sys

ROOT = "/opt/dlami/nvme/jasonleeeli"
D = os.path.dirname(os.path.abspath(__file__))
apply = "--apply" in sys.argv

chmod_n = same_n = absent_n = notreg_n = 0
for f in sorted(glob.glob(f"{D}/modes/*.modes.tsv")):
    for line in open(f, encoding="utf-8", errors="surrogateescape"):
        rel, mode = line.rstrip("\n").rsplit("\t", 1)
        p = os.path.join(ROOT, rel)
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            absent_n += 1
            continue
        if not stat.S_ISREG(st.st_mode):
            notreg_n += 1
            continue
        want = int(mode, 8)
        if stat.S_IMODE(st.st_mode) == want or want == 0:
            same_n += 1
            continue
        chmod_n += 1
        if apply:
            os.chmod(p, want)

link_n = exist_n = noparent_n = 0
for f in sorted(glob.glob(f"{D}/links/*.symlinks.tsv")):
    for line in open(f, encoding="utf-8", errors="surrogateescape"):
        if "\t" not in line:
            continue
        rel, target = line.rstrip("\n").split("\t", 1)
        p = os.path.join(ROOT, rel)
        if os.path.lexists(p):
            exist_n += 1
            continue
        if not os.path.isdir(os.path.dirname(p)):
            noparent_n += 1
            if apply:
                os.makedirs(os.path.dirname(p), exist_ok=True)
        link_n += 1
        if apply:
            os.symlink(target, p)

print(f"{'APPLIED' if apply else 'DRY RUN'}: chmod {chmod_n}, already right {same_n}, gone locally {absent_n}, "
      f"not regular {notreg_n}; symlinks created {link_n}, path exists {exist_n}, parent missing {noparent_n}")
