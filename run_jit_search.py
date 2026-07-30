import os
import sys

root = r"C:\Users\ericb\Documents\GitHub\PSD_opt"

# Suche nach JIT-relevanten Mustern in .py Dateien
jit_patterns = [
    'jit', 'njit', 'numba', '@nb', 'nb_', 
    'cython', 'cdef', 'cimport',
    'ctypes', 'ffi', 'from pbe_core'
]

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
                        # Prüfe auf JIT-Muster
                        for pattern in jit_patterns:
                            if pattern.lower() in line.lower():
                                relpath = os.path.relpath(filepath, root)
                                results.append((relpath, i, line.rstrip()))
                                break  # Nur einmal pro Zeile hinzufügen
            except Exception as e:
                print(f"Error reading {filepath}: {e}", file=sys.stderr)

# Sortiere nach Datei und Zeile
results.sort(key=lambda x: (x[0], x[1]))

output_lines = []
for filepath, lineno, line in results:
    output_lines.append(f"{filepath}:{lineno}: {line[:200]}")

result_text = '\n'.join(output_lines)
print(result_text)

# Write to file for easier reading
with open(os.path.join(root, 'jit_results.txt'), 'w', encoding='utf-8') as f:
    f.write(result_text)

print(f"\n\n{'='*60}")
print(f"Total: {len(results)} matches found")
print("Results also written to jit_results.txt")
