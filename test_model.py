"""Quick test script to verify model functionality"""
import torch
import sys
from model import ModelConfig, DynamicHybridModel

def test_model_basic():
    """Test basic model instantiation and forward pass"""
    print("Testing model instantiation...")

    # Create small config for quick testing
    config = ModelConfig()
    config.hidden_dim = 256
    config.num_layers = 2
    config.num_heads = 4
    config.batch_size = 2
    config.max_seq_len = 128
    config.gradient_checkpointing = False
    config.mixed_precision = False

    # Create model
    model = DynamicHybridModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits, outputs = model(input_ids)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch!"

    # Test with labels (training mode)
    print("\nTesting training mode with loss...")
    labels = input_ids.clone()
    model.train()
    logits, outputs = model(input_ids, labels=labels)
    loss = outputs['loss']

    print(f"✓ Training forward pass successful")
    print(f"  Loss: {loss.item():.4f}")

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print(f"✓ Backward pass successful")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

    return True

if __name__ == '__main__':
    try:
        test_model_basic()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
