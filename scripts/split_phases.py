import re, pathlib, sys
src = pathlib.Path(r'e:/Projects/Thesis/docs/project_logs/phases_01_to_09.md')
text = src.read_text(encoding='utf-8')
# Split on headings starting with ## and containing (Phase X)
parts = re.split(r'(?m)^##\s+.*?\(Phase\s+(\d+)\)', text)
# The split will produce ['', '9', 'content after phase9', '8', 'content after phase8', ...]
# We'll iterate pairs
it = iter(parts[1:])
for phase_num, content in zip(it, it):
    phase = int(phase_num)
    # prepend heading line
    heading_match = re.search(r'(?m)^##\s+.*?\(Phase\s+%d\).*' % phase, text)
    heading = heading_match.group(0) if heading_match else f'## Phase {phase}'
    full = heading + '\n' + content.strip() + '\n'
    out_path = pathlib.Path(r'e:/Projects/Thesis/docs/project_logs/phase_%d.md' % phase)
    out_path.write_text(full, encoding='utf-8')
print('done')
