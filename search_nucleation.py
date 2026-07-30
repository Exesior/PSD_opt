import os
import subprocess

root = r"C:\Users\ericb\Documents\GitHub\PSD_opt"

# Suche nach allen Vorkommen von "nucleation" in .py Dateien
results = []

for dirpath, dirnames, filenames in os.walk(root):
    # Skip __pycache__ directories
    dirnames[:] = [d for d in dirnames if d != '__pycache__']
    for filename in filenames:
        if filename.endswith('.py'):
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'nucleation' in line.lower() or '.step(' in line.lower():
                            results.append((filepath, i, line.rstrip()))
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# Sortiere nach Datei und Zeile
results.sort(key=lambda x: (x[0], x[1]))

for filepath, lineno, line in results:
    relpath = os.path.relpath(filepath, root)
    print(f"{relpath}:{lineno}: {line[:200]}")
