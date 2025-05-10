# fix_includes.py
import os
import re

def fix_includes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cpp', '.cu', '.h')):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # hyper_butterfly/ 접두사 제거
                original = content
                content = re.sub(r'#include\s*<hyper_butterfly/([^>]+)>', r'#include <\1>', content)
                content = re.sub(r'#include\s*"hyper_butterfly/([^"]+)"', r'#include "\1"', content)
                
                if content != original:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {filepath}")

# 실행
fix_includes('src')