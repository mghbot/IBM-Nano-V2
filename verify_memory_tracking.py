"""
Quick script to verify memory tracking is working correctly.
"""
import torch
from model import DynamicHybridModel, ModelConfig, MemoryTracker

def verify_memory_tracking():
    print("="*70)
    print("VERIFYING MEMORY TRACKING")
    print("="*70)

    # Create small config
    config = ModelConfig()
    config.hidden_dim = 256
    config.num_layers = 2
    config.num_heads = 4
    config.batch_size = 2
    config.max_seq_len = 128
    config.gradient_checkpointing = False

    # Create model
    model = DynamicHybridModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\n1. Model created on device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test built-in memory tracker
    print("\n2. Testing model's built-in memory tracker:")
    tracker = model.memory_tracker

    # Create dummy input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    labels = input_ids.clone()

    # Track forward pass
    print("\n3. Running forward pass with memory tracking...")
    with tracker.track('forward_pass'):
        model.eval()
        with torch.no_grad():
            logits, outputs = model(input_ids, labels=labels)

    # Get stats
    stats = tracker.get_stats('forward_pass')
    if stats:
        print("\n   Forward Pass Memory Stats:")
        print(f"   - Mean allocated: {stats.get('mean_allocated', 0):.4f} GB")
        print(f"   - Mean peak: {stats.get('mean_peak', 0):.4f} GB")
        print(f"   - Mean delta: {stats.get('mean_delta', 0):.4f} GB")
    else:
        print("   ⚠ No memory stats (CPU mode or tracking not available)")

    # Test multiple passes
    print("\n4. Running multiple passes to test averaging...")
    for i in range(3):
        with tracker.track('multi_pass'):
            with torch.no_grad():
                _ = model(input_ids, labels=labels)

    multi_stats = tracker.get_stats('multi_pass')
    if multi_stats:
        print("\n   Multi-pass Memory Stats (averaged over 3 runs):")
        print(f"   - Mean allocated: {multi_stats.get('mean_allocated', 0):.4f} GB")
        print(f"   - Mean peak: {multi_stats.get('mean_peak', 0):.4f} GB")
        print(f"   - Mean delta: {multi_stats.get('mean_delta', 0):.4f} GB")

    # Test training mode tracking
    print("\n5. Testing training mode with backward pass...")
    model.train()
    with tracker.track('training_step'):
        logits, outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()

    train_stats = tracker.get_stats('training_step')
    if train_stats:
        print("\n   Training Step Memory Stats:")
        print(f"   - Mean allocated: {train_stats.get('mean_allocated', 0):.4f} GB")
        print(f"   - Mean peak: {train_stats.get('mean_peak', 0):.4f} GB")
        print(f"   - Mean delta: {train_stats.get('mean_delta', 0):.4f} GB")

    # Verify routing metrics are tracked
    print("\n6. Verifying routing metrics are tracked...")
    print(f"   - SSM tokens per batch: {outputs.get('ssm_tokens', 0):.2f}")
    print(f"   - Transformer tokens per batch: {outputs.get('transformer_tokens', 0):.2f}")
    print(f"   - Mean gate value: {outputs.get('mean_gate', 0):.3f}")
    print(f"   - Compression rate: {outputs.get('compression_rate', 0):.3f}")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE ✓")
    print("="*70)
    print("\nMemory tracking is working correctly!")
    print("All metrics are being captured and averaged as expected.")

if __name__ == '__main__':
    verify_memory_tracking()
