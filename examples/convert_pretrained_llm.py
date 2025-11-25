import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reality_stone.models.llm_adapter import (
    RealityStoneLLMAdapter,
    LLMAdapterConfig,
    convert_pretrained_llm_to_reality_stone,
    finetune_adapted_llm
)


def example_1_basic_conversion():
    print("=" * 60)
    print("Example 1: Basic Conversion - GPT-2 to Reality Stone")
    print("=" * 60)
    
    config = LLMAdapterConfig(
        pretrained_model_name="gpt2",
        use_causal_lm=True,
        hidden_dim=768,
        num_hyperbolic_layers=4,
        use_bellman_coordinates=True,
        use_riemannian_metric=True,
        use_triple_hyperbolic=True,
        use_lagrangian=True,
        freeze_pretrained=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = convert_pretrained_llm_to_reality_stone("gpt2", config, device)
    
    print("\nModel architecture:")
    print(f"- Pretrained LLM: {config.pretrained_model_name}")
    print(f"- Hyperbolic layers: {config.num_hyperbolic_layers}")
    print(f"- Bellman coordinates: {config.use_bellman_coordinates}")
    print(f"- Riemannian metric: {config.use_riemannian_metric}")
    print(f"- Triple hyperbolic: {config.use_triple_hyperbolic}")
    print(f"- Lagrangian: {config.use_lagrangian}")
    
    test_input = "The quick brown fox"
    input_ids = model.tokenizer.encode(test_input, return_tensors="pt").to(device)
    
    print(f"\nTest input: '{test_input}'")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model.forward(input_ids, return_all=True)
    
    print("\nForward pass results:")
    print(f"- Logits shape: {outputs['logits'].shape if outputs['logits'] is not None else 'N/A'}")
    print(f"- Hidden states layers: {len(outputs['hidden_states'])}")
    print(f"- Metrics computed: {len(outputs['metrics'])}")
    print(f"- Velocities computed: {len(outputs['velocities'])}")
    print(f"- Lagrangian losses: {len(outputs['lagrangian_losses'])}")
    
    generated = model.generate(input_ids, max_length=30)
    generated_text = model.tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text: '{generated_text}'")
    
    print("\nConversion successful!")


def example_2_bert_conversion():
    print("\n" + "=" * 60)
    print("Example 2: BERT to Reality Stone (Encoder-only)")
    print("=" * 60)
    
    config = LLMAdapterConfig(
        pretrained_model_name="bert-base-uncased",
        use_causal_lm=False,
        hidden_dim=768,
        num_hyperbolic_layers=3,
        use_bellman_coordinates=True,
        use_riemannian_metric=True,
        use_triple_hyperbolic=True,
        freeze_pretrained=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convert_pretrained_llm_to_reality_stone("bert-base-uncased", config, device)
    
    test_input = "The quick brown fox jumps over the lazy dog."
    input_ids = model.tokenizer.encode(test_input, return_tensors="pt").to(device)
    
    print(f"\nTest input: '{test_input}'")
    
    with torch.no_grad():
        outputs = model.forward(input_ids, return_all=True)
    
    print("\nForward pass results:")
    print(f"- Final hidden shape: {outputs['final_hidden'].shape}")
    print(f"- Adapted hidden shape: {outputs['adapted_hidden'].shape if outputs['adapted_hidden'] is not None else 'N/A'}")
    print(f"- Metrics: {len(outputs['metrics'])}")
    
    print("\nBERT conversion successful!")


def example_3_selective_layer_insertion():
    print("\n" + "=" * 60)
    print("Example 3: Selective Layer Insertion")
    print("=" * 60)
    
    config = LLMAdapterConfig(
        pretrained_model_name="gpt2",
        use_causal_lm=True,
        hidden_dim=768,
        num_hyperbolic_layers=3,
        hyperbolic_insertion_positions=[2, 6, 10],
        use_bellman_coordinates=True,
        use_riemannian_metric=True,
        use_triple_hyperbolic=True,
        freeze_pretrained=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convert_pretrained_llm_to_reality_stone("gpt2", config, device)
    
    print(f"\nHyperbolic layers inserted at positions: {config.hyperbolic_insertion_positions}")
    print(f"Total transformer layers: {len(model.pretrained_llm.transformer.h)}")
    
    test_input = "Reality Stone transforms"
    input_ids = model.tokenizer.encode(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.forward(input_ids, return_all=True)
    
    print(f"\nMetrics computed at {len(outputs['metrics'])} positions")
    print(f"Velocities computed: {len(outputs['velocities'])}")
    
    print("\nSelective insertion successful!")


def example_4_compute_loss():
    print("\n" + "=" * 60)
    print("Example 4: Computing Loss with Reality Stone Components")
    print("=" * 60)
    
    config = LLMAdapterConfig(
        pretrained_model_name="gpt2",
        use_causal_lm=True,
        hidden_dim=768,
        num_hyperbolic_layers=2,
        use_lagrangian=True,
        lagrangian_weight=0.1,
        metric_regularization_weight=0.01,
        freeze_pretrained=False,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convert_pretrained_llm_to_reality_stone("gpt2", config, device)
    
    test_input = "The Reality Stone integrates Riemannian geometry"
    input_ids = model.tokenizer.encode(test_input, return_tensors="pt").to(device)
    labels = input_ids.clone()
    
    loss, loss_dict = model.compute_loss(input_ids, labels)
    
    print("\nLoss components:")
    print(f"- Total loss: {loss_dict['total']:.4f}")
    print(f"- Language modeling loss: {loss_dict['lm']:.4f}")
    print(f"- Lagrangian loss: {loss_dict['lagrangian']:.4f}")
    print(f"- Metric regularization: {loss_dict['metric_reg']:.4f}")
    
    print("\nLoss computation successful!")


def example_5_parameter_analysis():
    print("\n" + "=" * 60)
    print("Example 5: Parameter Analysis")
    print("=" * 60)
    
    config = LLMAdapterConfig(
        pretrained_model_name="gpt2",
        hidden_dim=768,
        num_hyperbolic_layers=4,
        freeze_pretrained=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convert_pretrained_llm_to_reality_stone("gpt2", config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\nParameter statistics:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Frozen parameters: {frozen_params:,}")
    print(f"- Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    reality_stone_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'pretrained_llm' not in n
    )
    
    print(f"\nReality Stone components: {reality_stone_params:,} parameters")
    print(f"Original LLM: {total_params - reality_stone_params:,} parameters")
    
    print("\nParameter analysis complete!")


def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "Reality Stone LLM Adapter Examples")
    print("=" * 80)
    
    try:
        example_1_basic_conversion()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_2_bert_conversion()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_3_selective_layer_insertion()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_4_compute_loss()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_5_parameter_analysis()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()


