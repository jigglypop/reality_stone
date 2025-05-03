import os

def generate_markdown_for_csrc(root_dir):
    markdown = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if filename.endswith(('.h', '.cpp', '.cu', '.py')):
                filepath = os.path.join(dirpath, filename)
                relpath = os.path.relpath(filepath, root_dir)
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                markdown.append(f"### `{relpath}`\n```cpp\n{code}\n```\n")
    return "\n".join(markdown)

csrc_dir = "hyper_butterfly"
markdown_output = generate_markdown_for_csrc(csrc_dir)

# Save to a file
output_path = './test.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(markdown_output)

print(f"Markdown for csrc files written to {output_path}")
