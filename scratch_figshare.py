import os
import requests
import zipfile
import io

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size, block_size=1024*1024):
        self.url = url
        self.size = size
        self.position = 0
        self.block_size = block_size
        self.cache = {}

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        return self.position

    def tell(self):
        return self.position

    def _read_block(self, block_index):
        if block_index in self.cache:
            return self.cache[block_index]
        
        start = block_index * self.block_size
        end = min(start + self.block_size - 1, self.size - 1)
        headers = {"Range": f"bytes={start}-{end}"}
        resp = requests.get(self.url, headers=headers)
        resp.raise_for_status()
        data = resp.content
        self.cache[block_index] = data
        return data

    def read(self, size=-1):
        if size == -1:
            size = self.size - self.position
        if size == 0:
            return b""
            
        data = bytearray()
        while size > 0 and self.position < self.size:
            block_index = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._read_block(block_index)
            
            chunk_size = min(size, len(block) - block_offset)
            data.extend(block[block_offset:block_offset + chunk_size])
            
            self.position += chunk_size
            size -= chunk_size
            
        return bytes(data)

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def readable(self):
        return True

    def seekable(self):
        return True

url = "https://ndownloader.figshare.com/files/27039812"
with requests.get(url, stream=True, allow_redirects=True) as resp:
    size = int(resp.headers.get("Content-Length", 0))
    resolved_url = resp.url

f = HTTPRangeFile(resolved_url, size)
z = zipfile.ZipFile(f)

for c in z.namelist():
    if "Subject_1001" in c or "Round_2" in c:
        print(c)
