import sys
try:
    import reality_stone._rust
    print("Import Successful")
    print(dir(reality_stone._rust))
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

