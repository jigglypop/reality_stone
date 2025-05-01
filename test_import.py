import sys
import os

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # hyper_butterfly 임포트 테스트
    import hyper_butterfly
    print("✅ hyper_butterfly 임포트 성공!")
    print(f"모듈 경로: {hyper_butterfly.__file__}")
    
    # 사용 가능한 함수들 확인
    print("\n사용 가능한 함수들:")
    for name in dir(hyper_butterfly):
        if not name.startswith('_'):
            print(f"- {name}")
    
    # C++ 확장 모듈 테스트
    try:
        import hyper_butterfly._C
        print("\n✅ C++ 확장 모듈 임포트 성공!")
    except ImportError as e:
        print(f"\n❌ C++ 확장 모듈 임포트 실패: {e}")

except ImportError as e:
    print(f"❌ hyper_butterfly 임포트 실패: {e}")
    
    # 디버깅 정보 출력
    print("\nPython 경로:")
    for p in sys.path:
        print(f"- {p}")
    
    print(f"\n현재 디렉토리: {os.getcwd()}")
    
    try:
        import hyper_butterfly
        print("\n✅ core 모듈은 임포트 가능합니다.")
    except ImportError as e:
        print(f"\n❌ core 모듈도 임포트 불가능: {e}") 