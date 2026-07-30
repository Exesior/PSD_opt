import os
import re

search_dir = r"C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe"

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if '_prepare_break_config' in line:
                            print(f'{os.path.relpath(filepath, search_dir)}:{i}: {line.strip()}')
            except Exception as e:
                pass
