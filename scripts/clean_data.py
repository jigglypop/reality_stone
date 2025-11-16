"""데이터 클렌징 - 위키 마크업 및 중복 제거"""
import re

input_file = 'tests/data/text.txt'
output_file = 'tests/data/text_cleaned.txt'

print(f"[INFO] 읽기: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"[INFO] 원본 길이: {len(text):,} chars")

# 위키 마크업 제거
text = re.sub(r'formula_\d+', '', text)  # formula_1, formula_2 등
text = re.sub(r'\{\{[^}]+\}\}', '', text)  # {{...}} 템플릿
text = re.sub(r'\[\[[^\]]+\]\]', '', text)  # [[...]] 링크
text = re.sub(r'[:#\*\{\}]', '', text)  # 위키 문법 기호
text = re.sub(r'\([A-Za-z\s,]+\)', '', text)  # 영어 괄호
text = re.sub(r'\s+', ' ', text)  # 연속 공백 제거

print(f"[INFO] 클렌징 후 길이: {len(text):,} chars")

# 저장
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"[INFO] 저장: {output_file}")

