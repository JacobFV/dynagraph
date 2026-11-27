"""Extract every COMPLETE member from a (possibly still-downloading) .tar.gz prefix (owner: agent EXT).
Used to start validating on the first sequences of a large official archive before the download ends.
Members are written under <out>; existing files of the right size are skipped. Paths are sanitised (no absolute
paths, no '..'). usage: python -I tar_prefix_extract.py <archive.tar.gz[.part]> <out_dir>"""
import sys
import tarfile
import zlib
from pathlib import Path


class GzPrefix:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.buf = b""

    def read(self, n=-1):
        while n < 0 or len(self.buf) < n:
            c = self.f.read(1 << 20)
            if not c:
                break
            self.buf += self.d.decompress(c)
        out, self.buf = (self.buf, b"") if n < 0 else (self.buf[:n], self.buf[n:])
        return out


def extract(arc, out):
    out = Path(out).resolve()
    n = 0
    tf = tarfile.open(fileobj=GzPrefix(arc), mode="r|")
    try:
        for m in tf:
            dst = (out / m.name).resolve()
            if not str(dst).startswith(str(out)) or m.issym() or m.islnk():
                continue
            if m.isdir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            if len(data) != m.size:
                break
            if not (dst.exists() and dst.stat().st_size == m.size):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
            n += 1
    except (EOFError, tarfile.ReadError, zlib.error):
        pass
    print(f"{n} complete members", flush=True)


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
