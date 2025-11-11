import torch.nn as nn

def convert_to_hyperbolic(model: nn.Module, layer_map: dict, **kwargs):
    """
    모델의 레이어를 지정된 하이퍼볼릭 레이어로 재귀적으로 변환합니다.

    Args:
        model (nn.Module): 변환할 PyTorch 모델.
        layer_map (dict): 변환할 레이어 타입과 새로운 레이어 클래스를 매핑한 딕셔너리.
                          예: {nn.Linear: EquivalentHyperbolicLinear}
        **kwargs: 새로운 레이어 클래스의 생성자에 전달될 추가 인자.
    """
    for name, module in model.named_children():
        # 재귀적으로 하위 모듈 탐색
        if len(list(module.children())) > 0:
            convert_to_hyperbolic(module, layer_map, **kwargs)

        # 맵에 정의된 레이어 타입이면 변환 수행
        replaced = False
        for old_layer_type, new_layer_class in layer_map.items():
            if isinstance(module, old_layer_type):
                new_layer = new_layer_class.from_linear(module, **kwargs)
                setattr(model, name, new_layer)
                try:
                    print(f"Replaced '{name}' -> {new_layer_class.__name__}")
                except Exception:
                    pass
                replaced = True
                break  # 한 번 변환되면 루프 종료

        # HuggingFace GPT 계열의 Conv1D 처리 (유형명이 문자열로만 구분되는 경우)
        if not replaced and 'Conv1D' in str(type(module)):
            # layer_map의 첫 번째 타깃 클래스를 사용
            if layer_map:
                new_layer_class = list(layer_map.values())[0]
                new_layer = new_layer_class.from_linear(module, **kwargs)
                setattr(model, name, new_layer)
                try:
                    print(f"Replaced Conv1D '{name}' -> {new_layer_class.__name__}")
                except Exception:
                    pass