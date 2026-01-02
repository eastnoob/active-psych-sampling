"""Backup and prune run directories: create a zip of all run dirs except the one to keep and backups/, then delete those dirs.
Usage: python scripts/backup_and_prune_runs.py --base <runs_path> --keep <dir_to_keep>
"""
import argparse
from pathlib import Path
import zipfile
import shutil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='KEY_CESHI/simulate_subjects/linear/runs')
parser.add_argument('--keep', default='50_likert6_mean225_475')
args = parser.parse_args()

base = Path(args.base)
keep = args.keep
backup_dir = base / 'backups'
backup_dir.mkdir(parents=True, exist_ok=True)

items = [p for p in base.iterdir() if p.is_dir() and p.name not in (keep, 'backups')]
if not items:
    print('No items to backup; nothing to do')
else:
    ts = time.strftime('%Y%m%d%H%M%S')
    zip_path = backup_dir / f'runs_backup_{ts}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in items:
            for f in p.rglob('*'):
                z.write(f, arcname=str(p.name / f.relative_to(p)))
    print('Created backup:', zip_path)

    # Verify zip exists then delete directories
    if zip_path.exists():
        for p in items:
            print('Removing:', p)
            shutil.rmtree(p)
        print('Deleted old runs')
    else:
        print('Backup failed; aborting deletion')

print('Remaining run dirs:')
for p in base.iterdir():
    if p.is_dir():
        print(' -', p.name)
