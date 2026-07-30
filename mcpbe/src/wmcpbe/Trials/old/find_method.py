"""Find create_nucleation_handler method."""

with open('wmcpbe/mcpbe_base.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
idx = content.find('def create_nucleation_handler')
if idx >= 0:
    # Find next method definition
    next_def = content.find('\n    def ', idx + 10)
    if next_def < 0:
        next_def = len(content)
    with open('wmcpbe/Trials/method_output.txt', 'w', encoding='utf-8') as out:
        out.write(content[idx:next_def])
    print("Method written to method_output.txt")
else:
    print("Not found")
