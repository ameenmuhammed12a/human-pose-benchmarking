"""Read selected members out of a huge remote ZIP over HTTP range requests."""
import io, os, sys, zipfile, urllib.request

class HttpRangeFile:
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self._pos, self._cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("server does not support range requests")

    def seekable(self): return True
    def writable(self): return False
    def close(self): pass
    @property
    def closed(self): return False
    def readable(self): return True
    def tell(self): return self._pos

    def seek(self, off, whence=0):
        self._pos = off if whence == 0 else (self._pos + off if whence == 1 else self.size + off)
        return self._pos

    def _fetch(self, start, end):
        """Inclusive byte range, served from a block cache."""
        out = bytearray()
        b0, b1 = start // self.block, end // self.block
        for b in range(b0, b1 + 1):
            if b not in self._cache:
                s = b * self.block
                e = min(s + self.block, self.size) - 1
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={s}-{e}"})
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(req, timeout=120) as r:
                            self._cache[b] = r.read()
                        break
                    except Exception:
                        if attempt == 3: raise
            out += self._cache[b]
        off = start - b0 * self.block
        return bytes(out[off:off + (end - start + 1)])

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0: return b""
        d = self._fetch(self._pos, self._pos + n - 1)
        self._pos += len(d)
        return d

def open_remote_zip(url, block=1 << 20):
    return zipfile.ZipFile(HttpRangeFile(url, block=block))


def read_member_at(fp, header_offset):
    """Read one archive member given only its local-header offset.

    This is what makes a cached offset worth having: the 22.5 MB central directory never has
    to be read again, because a ZIP local file header carries everything needed to extract
    the member sitting behind it.
    """
    import struct, zlib
    head = fp._fetch(header_offset, header_offset + 29)
    sig, _ver, _flags, method, _t, _d, _crc, csize, _usize, nlen, elen = \
        struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise ValueError(f"no local file header at offset {header_offset}")
    start = header_offset + 30 + nlen + elen
    if csize in (0, 0xFFFFFFFF):
        # streamed or zip64 sizes live in the extra field / data descriptor, which this
        # fast path does not parse; the caller should fall back to the full index
        raise ValueError("member size not in the local header (zip64 or streamed)")
    raw = fp._fetch(start, start + csize - 1)
    if method == 0:
        return raw
    if method == 8:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported compression method {method}")
