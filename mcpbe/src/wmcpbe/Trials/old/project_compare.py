"""
Project Comparison: Current vs Backup.

This script compares the current wmcpbe project with its backup
to identify:
1. New files added
2. Files removed
3. Modified files (by size/timestamp)
4. Potential optimization opportunities
"""

import os
import sys
from pathlib import Path
from datetime import datetime

CURRENT_DIR = r"C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe"
BACKUP_DIR = r"C:\Users\ericb\Documents\GitHub\PSD_opt - backup\mcpbe\src\wmcpbe"


def get_all_files(base_dir: str) -> dict:
    """Get all files in directory with metadata."""
    files = {}
    base = Path(base_dir)
    
    for path in base.rglob("*"):
        if path.is_file():
            rel_path = str(path.relative_to(base))
            try:
                stat = path.stat()
                files[rel_path] = {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'path': str(path),
                }
            except Exception as e:
                files[rel_path] = {'size': 0, 'modified': None, 'path': str(path), 'error': str(e)}
    
    return files


def compare_projects():
    """Compare current and backup directories."""
    print("=" * 80)
    print("PROJECT COMPARISON: Current vs Backup")
    print("=" * 80)
    print(f"\nCurrent: {CURRENT_DIR}")
    print(f"Backup:  {BACKUP_DIR}")
    
    current_files = get_all_files(CURRENT_DIR)
    backup_files = get_all_files(BACKUP_DIR)
    
    current_set = set(current_files.keys())
    backup_set = set(backup_files.keys())
    
    # New files (in current, not in backup)
    new_files = current_set - backup_set
    
    # Removed files (in backup, not in current)
    removed_files = backup_set - current_set
    
    # Common files (potentially modified)
    common_files = current_set & backup_set
    
    print("\n" + "=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    print(f"  Current files: {len(current_files)}")
    print(f"  Backup files:  {len(backup_files)}")
    print(f"  New files:     {len(new_files)}")
    print(f"  Removed files: {len(removed_files)}")
    print(f"  Common files:  {len(common_files)}")
    
    # Show new files
    if new_files:
        print("\n" + "-" * 80)
        print("NEW FILES (in current, not in backup):")
        print("-" * 80)
        for f in sorted(new_files):
            info = current_files[f]
            size_kb = info['size'] / 1024
            print(f"  [+] {f} ({size_kb:.1f} KB)")
    
    # Show removed files
    if removed_files:
        print("\n" + "-" * 80)
        print("REMOVED FILES (in backup, not in current):")
        print("-" * 80)
        for f in sorted(removed_files):
            info = backup_files[f]
            size_kb = info['size'] / 1024 if info['size'] > 0 else 0
            print(f"  [-] {f} ({size_kb:.1f} KB)")
    
    # Show modified files (size changed by >5%)
    modified = []
    for f in common_files:
        curr_size = current_files[f]['size']
        back_size = backup_files[f]['size']
        if back_size > 0:
            change = abs(curr_size - back_size) / back_size
            if change > 0.05:  # >5% change
                diff = curr_size - back_size
                modified.append((f, curr_size, back_size, diff))
    
    if modified:
        print("\n" + "-" * 80)
        print("MODIFIED FILES (size changed >5%):")
        print("-" * 80)
        for f, curr, back, diff in sorted(modified, key=lambda x: abs(x[3]), reverse=True):
            change_pct = (diff / back * 100) if back > 0 else 0
            sign = "+" if diff > 0 else ""
            print(f"  [~] {f}")
            print(f"      Current: {curr/1024:.1f} KB, Backup: {back/1024:.1f} KB ({sign}{change_pct:.1f}%)")
    
    return {
        'new': sorted(new_files),
        'removed': sorted(removed_files),
        'modified': [(f, c, b, d) for f, c, b, d in sorted(modified, key=lambda x: abs(x[3]), reverse=True)],
        'current_files': current_files,
        'backup_files': backup_files,
    }


if __name__ == "__main__":
    results = compare_projects()
    
    # Save detailed report
    report_path = os.path.join(CURRENT_DIR, "Trials", "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PROJECT COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("NEW FILES:\n")
        for file in results['new']:
            f.write(f"  + {file}\n")
        
        f.write("\nREMOVED FILES:\n")
        for file in results['removed']:
            f.write(f"  - {file}\n")
        
        f.write("\nMODIFIED FILES:\n")
        for file, curr, back, diff in results['modified']:
            pct = (diff / back * 100) if back > 0 else 0
            f.write(f"  ~ {file} ({pct:+.1f}%)\n")
    
    print(f"\n\nDetailed report saved to: {report_path}")
