"""Check for remaining legacy references in wmcpbe module."""
import os

files_to_check = [
    'mcpbe/src/wmcpbe/mcpbe_post.py',
    'mcpbe/src/wmcpbe/mcpbe_nucleation.py',
    'mcpbe/src/wmcpbe/mcpbe_compression.py',
    'mcpbe/src/wmcpbe/fenwick_new.py',
    'mcpbe/src/wmcpbe/reconstruction_mixin.py',
]

legacy_terms = ['COLEVAL', 'BREAKRVAL', 'CORR_BETA', 'BREAKFVAL', '_kb_', 'nb_']

print('LEGACY REFERENCE CHECK')
print('=' * 70)

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f'{filepath}: FILE NOT FOUND')
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    found = []
    for term in legacy_terms:
        count = content.count(term)
        if count > 0:
            found.append(f'{term}: {count}')
    
    if found:
        print(f'{filepath}:')
        for item in found:
            print(f'  ⚠️  {item}')
    else:
        print(f'{filepath}: ✅ CLEAN')

print('=' * 70)
