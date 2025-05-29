import numpy as np
import struct
import mmap
import os


# TODO: pass mmap.MADV_RANDOM as argument
class BitboardDecoder():
    def __init__(self, path, memory_mapped=True, length = 0, pv_depth = 0, prefer_sequential = False):
        self.struct_format = '>QQQQQQQQQQQQbbbe'
        self.path = path
        self.memory_mapped = memory_mapped
        self.pv_depth = pv_depth

        self.f = open(self.path, "rb", buffering=0)

        if self.pv_depth > 0:
            # We could write f'{pv_depth}H' but for consistency with the encoding phase we leave it like this:
            self.struct_format += pv_depth * 'H' # H is an unsigned short (2 bytes), these fields hold pv moves

        self.line_size = struct.calcsize(self.struct_format)
        self._length = length if length > 0 else os.path.getsize(self.path) // self.line_size

        if self.memory_mapped:
            self.mm: mmap.mmap = self._open_map()
            self.mm.madvise(
                mmap.MADV_SEQUENTIAL if prefer_sequential else mmap.MADV_RANDOM,
                0, self._length *
                    self.line_size
            )

    def _open_map(self):
        return mmap.mmap(self.f.fileno(), self.line_size * self._length, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)

    def read_line(self, idx):
        offset = idx*self.line_size

        if self.memory_mapped:
            bytes = self.mm[offset:offset + self.line_size]
        else:
            self.f.seek(offset)
            bytes = self.f.read(self.line_size)

        try:
            line = struct.unpack(self.struct_format, bytes)
        except Exception as e:
            print(f'Exception thrown while decoding: {idx}')
            print(e)
            exit(1)

        return line

    def free(self, start, end):
        self.mm.madvise(mmap.MADV_DONTNEED, start * self.line_size, end * self.line_size)

    def prefetch(self, start, end):
        self.mm.madvise(mmap.MADV_WILLNEED, start * self.line_size, end * self.line_size)

    def reopen(self, idx):
        self.mm.close()

    def length(self):
        return self._length



