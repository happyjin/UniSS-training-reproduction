"""Read unix modes from the central directory of each S3 zip volume via ranged GETs.
Writes <out>/<part>.modes.tsv : path<TAB>octal mode. Read-only on S3; writes only to <out>."""
import os, struct, subprocess, sys, tempfile

BUCKET = "stevehguo-ap-southeast-3"
PREFIX = "speech_model_backup/jasonleeeli/jasonleeeli_backup_zip/"
out = sys.argv[1]
parts = sys.argv[2:]


def rng(key, start, end, tmp):
    subprocess.run(["aws", "s3api", "get-object", "--region", "ap-southeast-3", "--bucket", BUCKET,
                    "--key", key, "--range", f"bytes={start}-{end}", tmp],
                   check=True, stdout=subprocess.DEVNULL)
    with open(tmp, "rb") as f:
        return f.read()


def size_of(key):
    r = subprocess.run(["aws", "s3api", "head-object", "--region", "ap-southeast-3", "--bucket", BUCKET,
                        "--key", key, "--query", "ContentLength", "--output", "text"],
                       check=True, capture_output=True, text=True)
    return int(r.stdout.strip())


for part in parts:
    key = f"{PREFIX}jasonleeeli_backup.{part}.zip"
    dst = os.path.join(out, f"{part}.modes.tsv")
    if os.path.exists(dst):
        continue
    tmp = tempfile.mktemp(dir=out)
    size = size_of(key)
    tail_len = min(size, 1 << 20)
    tail = rng(key, size - tail_len, size - 1, tmp)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = rng(key, cd_off, cd_off + cd_size - 1, tmp)
    os.remove(tmp)
    n, p = 0, 0
    with open(dst + ".tmp", "w") as w:
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            made_by, = struct.unpack("<H", cd[p + 4:p + 6])
            nlen, xlen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            ext, = struct.unpack("<I", cd[p + 38:p + 42])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "surrogateescape")
            if made_by >> 8 == 3:  # unix
                w.write(f"{name}\t{(ext >> 16) & 0o7777:o}\n")
                n += 1
            p += 46 + nlen + xlen + clen
    os.rename(dst + ".tmp", dst)
    print(part, n, "entries", flush=True)
