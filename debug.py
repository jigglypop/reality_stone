import os
import subprocess
import sys

# _C.pyd 파일 찾기
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == '_C.pyd':
            pyd_path = os.path.join(root, file)
            print(f"Found _C.pyd at: {pyd_path}")
            
            # dumpbin으로 의존성 확인 (Visual Studio 도구)
            try:
                result = subprocess.run(['dumpbin', '/dependents', pyd_path], 
                                      capture_output=True, text=True)
                print("Dependencies:")
                print(result.stdout)
            except FileNotFoundError:
                print("dumpbin not found. Install Visual Studio tools.")