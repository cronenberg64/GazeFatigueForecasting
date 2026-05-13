import os
import requests
import zipfile
import io
import re
from collections import defaultdict
from tqdm import tqdm
import sys

def print_flush(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, original_url, url, size, block_size=1024*1024):
        self.original_url = original_url
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
        
        try:
            resp = requests.get(self.url, headers=headers)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                # print_flush("URL expired. Refreshing...")
                with requests.get(self.original_url, stream=True, allow_redirects=True) as new_resp:
                    new_resp.raise_for_status()
                    self.url = new_resp.url
                resp = requests.get(self.url, headers=headers)
                resp.raise_for_status()
            else:
                raise

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

def main():
    original_url = "https://ndownloader.figshare.com/files/27039812"
    
    print_flush("Fetching file size and resolving redirects...")
    with requests.get(original_url, stream=True, allow_redirects=True) as resp:
        resp.raise_for_status()
        size = int(resp.headers.get("Content-Length", 0))
        resolved_url = resp.url
        if size == 0:
            print_flush("Failed to get Content-Length!")
            sys.exit(1)
            
    print_flush(f"File size: {size / (1024**3):.2f} GB")
    print_flush(f"Resolved URL: {resolved_url.split('?')[0]}...")

    f = HTTPRangeFile(original_url, resolved_url, size, block_size=2*1024*1024) # 2MB cache blocks
    
    print_flush("Reading ZIP central directory (this may take a few seconds)...")
    z = zipfile.ZipFile(f)
    
    all_files = z.namelist()
    print_flush(f"Found {len(all_files)} files in the archive.")
    
    zip_files = [x for x in all_files if x.endswith('.zip')]
    
    subject_rounds = defaultdict(set)
    subject_files = defaultdict(list)
    
    # Pattern: Round_1/Subject_1001.zip
    pattern = re.compile(r'Round_(\d+)/Subject_\d(?P<subject>\d{3})\.zip')
    
    for filename in zip_files:
        match = pattern.search(filename)
        if match:
            round_num = match.group(1)
            subject = match.group('subject')
            subject_rounds[subject].add(round_num)
            subject_files[subject].append(filename)
            
    # Find subjects present in multiple rounds
    multi_round_subjects = [sub for sub, rounds in subject_rounds.items() if len(rounds) > 1]
    multi_round_subjects = sorted(multi_round_subjects)
    
    print_flush(f"Found {len(multi_round_subjects)} subjects in multiple rounds.")
    
    # Select 10 subjects
    selected_subjects = multi_round_subjects[:10]
    print_flush(f"Selected 10 subjects: {selected_subjects}")
    
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    
    downloaded_files = 0
    extracted_csvs = 0
    for subject in tqdm(selected_subjects, desc="Downloading subjects"):
        files_to_download = subject_files[subject]
        for filename in files_to_download:
            try:
                inner_zip_bytes = z.read(filename)
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_z:
                    csv_names = [name for name in inner_z.namelist() if name.endswith('.csv')]
                    for csv_name in csv_names:
                        basename = os.path.basename(csv_name)
                        local_path = os.path.join(data_dir, basename)
                        if not os.path.exists(local_path):
                            with open(local_path, 'wb') as lf:
                                lf.write(inner_z.read(csv_name))
                            extracted_csvs += 1
                downloaded_files += 1
            except Exception as e:
                print_flush(f"Error processing {filename}: {e}")

    print_flush(f"Successfully downloaded {downloaded_files} zip files and extracted {extracted_csvs} CSV files.")

    with open(os.path.join(data_dir, "README_data.md"), "w") as rf:
        rf.write("# GazeBase Data Subset\n\n")
        rf.write("This directory contains a subset of the GazeBase dataset downloaded via Figshare.\n")
        rf.write("Only the following 10 subjects (who appear in multiple rounds) were downloaded to serve as a sample for exploration:\n")
        rf.write(", ".join(selected_subjects) + "\n")
        
if __name__ == "__main__":
    main()
