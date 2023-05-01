import numpy as np
import pandas as pd
import struct
import mmap
import itertools
import os

STRUCT_FORMAT = '>QQQQQQQQQQQQbbbe'

class BitboardDecoder():
    def __init__(self, path, memory_mapped=True, length = 0):
        self.path = path
        self.memory_mapped = memory_mapped
        self.line_size = struct.calcsize(STRUCT_FORMAT)
        self._length = length if length > 0 else os.path.getsize(self.path) // self.line_size

        self.f = open(self.path, "rb", buffering=0)

        if self.memory_mapped:
            self.mm: mmap.mmap = self._open_map()
            self.mm.madvise(mmap.MADV_SEQUENTIAL, 0, self._length * self.line_size)

    def _open_map(self):
        return mmap.mmap(self.f.fileno(), self.line_size * self._length, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)

    def read_line(self, idx):
        offset = idx*self.line_size

        if self.memory_mapped:
            bytes = self.mm[offset:offset + self.line_size]
        else:
            self.f.seek(offset)
            bytes = self.f.read(self.line_size)

        line = np.array(struct.unpack(STRUCT_FORMAT, bytes))

        return line

    def free(self, start, end):
        self.mm.madvise(mmap.MADV_DONTNEED, start * self.line_size, end * self.line_size)

    def prefetch(self, start, end):
        self.mm.madvise(mmap.MADV_WILLNEED, start * self.line_size, end * self.line_size)

    def reopen(self, idx):
        self.mm.close()

    def length(self):
        return self._length



