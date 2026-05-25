import os
import shutil
import glob
import subprocess

# Move txt files
txt_files = glob.glob('data/*.txt')
for f in txt_files:
    shutil.move(f, 'data/pdfs/')

# Remove empty dirs
for d in ['data/raw', 'data/staged', 'data/curated']:
    if os.path.exists(d):
        os.rmdir(d)

if os.path.exists('data/pdfs/h.txt'):
    os.remove('data/pdfs/h.txt')

# Run ingestion script
subprocess.run(['python', 'agents/vector_store.py'], check=True)
