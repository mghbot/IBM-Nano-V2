"""
Benchmark script to compare DynamicHybridModel with IBM Granite Nano 350M.

Usage:
    # Fast testing with tiny config
    python benchmark.py --config tiny

    # Other config presets
    python benchmark.py --config small
    python benchmark.py --config medium
    python benchmark.py --config large

    # Custom parameters
    python benchmark.py --model dynamic --device cuda
    python benchmark.py --model granite --granite-path ~/models/granite-350m/
    python benchmark.py --config tiny --batch-size 4  # Override preset values

Available configs: tiny, small, medium, large
"""

import torch
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from collections import defaultdict

from model import DynamicHybridModel, ModelConfig, MemoryTracker

# =============================================================================
# CONFIG PRESETS
# =============================================================================

CONFIG_PRESETS = {
    'tiny': {
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 4,
        'batch_size': 2,
        'max_seq_len': 128,
        'description': 'Tiny config for fast testing'
    },
    'small': {
        'hidden_dim': 384,
        'num_layers': 6,
        'num_heads': 6,
        'batch_size': 4,
        'max_seq_len': 512,
        'description': 'Small config for quick experiments'
    },
    'medium': {
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'batch_size': 8,
        'max_seq_len': 1024,
        'description': 'Medium config for standard benchmarking'
    },
    'large': {
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'batch_size': 16,
        'max_seq_len': 2048,
        'description': 'Large config (default model size)'
    }
}

def apply_config_preset(config: ModelConfig, preset_name: str) -> ModelConfig:
    """Apply a configuration preset to a ModelConfig object"""
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown config preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}")

    preset = CONFIG_PRESETS[preset_name]
    print(f"\n📋 Applying config preset: '{preset_name}'")
    print(f"   {preset['description']}")

    for key, value in preset.items():
        if key != 'description' and hasattr(config, key):
            setattr(config, key, value)
            print(f"   {key}: {value}")

    return config

# =============================================================================
# GRANITE NANO MODEL WRAPPER
# =============================================================================

class GraniteNanoWrapper:
    """Wrapper for loading and running IBM Granite Nano models"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading Granite Nano from {model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map=device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            print(f"✓ Granite Nano loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Granite Nano: {e}")
            self.model = None
            self.tokenizer = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass"""
        if self.model is None:
            raise ValueError("Model not loaded")

        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def benchmark_forward_pass(
    model,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_iterations: int = 10,
    warmup: int = 3,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Benchmark forward pass performance"""

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device_obj)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        if isinstance(model, GraniteNanoWrapper):
            _ = model.forward(input_ids)
        else:
            with torch.no_grad():
                _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    print(f"  Running benchmark ({num_iterations} iterations)...")
    times = []
    memory_peaks = []

    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start = time.time()

        if isinstance(model, GraniteNanoWrapper):
            _ = model.forward(input_ids)
        else:
            with torch.no_grad():
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            memory_peaks.append(peak_mem)

    # Calculate statistics
    times = np.array(times)
    throughput = batch_size * seq_length / np.mean(times)

    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput_tokens_per_sec': throughput,
        'mean_memory_gb': np.mean(memory_peaks) if memory_peaks else 0,
        'peak_memory_gb': np.max(memory_peaks) if memory_peaks else 0,
    }

    return results

def benchmark_memory_scaling(
    model,
    batch_size: int,
    vocab_size: int,
    device: str = 'cuda',
    seq_lengths: list = [128, 256, 512, 1024, 2048]
) -> Dict[int, Dict[str, float]]:
    """Benchmark memory usage across different sequence lengths"""

    results = {}

    for seq_len in seq_lengths:
        print(f"\n  Benchmarking sequence length: {seq_len}")
        try:
            bench_results = benchmark_forward_pass(
                model, batch_size, seq_len, vocab_size,
                num_iterations=5, warmup=2, device=device
            )
            results[seq_len] = bench_results

            print(f"    Time: {bench_results['mean_time']:.3f}s")
            print(f"    Memory: {bench_results['mean_memory_gb']:.2f}GB")
            print(f"    Throughput: {bench_results['throughput_tokens_per_sec']:.0f} tok/s")
        except RuntimeError as e:
            print(f"    ⚠ Failed at seq_len={seq_len}: {e}")
            break

    return results

# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_models(
    dynamic_model: DynamicHybridModel,
    granite_model: Optional[GraniteNanoWrapper],
    config: ModelConfig,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Compare DynamicHybridModel with Granite Nano"""

    print("\n" + "="*70)
    print("BENCHMARKING: DynamicHybridModel")
    print("="*70)

    # Benchmark DynamicHybridModel
    dynamic_results = benchmark_forward_pass(
        dynamic_model,
        batch_size=config.batch_size,
        seq_length=config.max_seq_len,
        vocab_size=config.vocab_size,
        num_iterations=10,
        device=device
    )

    dynamic_params = sum(p.numel() for p in dynamic_model.parameters())

    print(f"\nDynamicHybridModel Results:")
    print(f"  Parameters: {dynamic_params / 1e6:.1f}M")
    print(f"  Forward time: {dynamic_results['mean_time']:.3f} ± {dynamic_results['std_time']:.3f}s")
    print(f"  Memory: {dynamic_results['mean_memory_gb']:.2f}GB")
    print(f"  Throughput: {dynamic_results['throughput_tokens_per_sec']:.0f} tokens/sec")

    comparison = {
        'dynamic': {
            'params': dynamic_params,
            'results': dynamic_results
        }
    }

    # Benchmark Granite Nano if available
    if granite_model and granite_model.model is not None:
        print("\n" + "="*70)
        print("BENCHMARKING: Granite Nano 350M")
        print("="*70)

        granite_results = benchmark_forward_pass(
            granite_model,
            batch_size=config.batch_size,
            seq_length=config.max_seq_len,
            vocab_size=config.vocab_size,
            num_iterations=10,
            device=device
        )

        granite_params = granite_model.get_num_parameters()

        print(f"\nGranite Nano Results:")
        print(f"  Parameters: {granite_params / 1e6:.1f}M")
        print(f"  Forward time: {granite_results['mean_time']:.3f} ± {granite_results['std_time']:.3f}s")
        print(f"  Memory: {granite_results['mean_memory_gb']:.2f}GB")
        print(f"  Throughput: {granite_results['throughput_tokens_per_sec']:.0f} tokens/sec")

        comparison['granite'] = {
            'params': granite_params,
            'results': granite_results
        }

        # Calculate improvements
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)

        speed_improvement = granite_results['mean_time'] / dynamic_results['mean_time']
        memory_reduction = 1 - (dynamic_results['mean_memory_gb'] / granite_results['mean_memory_gb'])

        print(f"\nDynamicHybridModel vs Granite Nano:")
        print(f"  Speed: {speed_improvement:.2f}x {'faster' if speed_improvement > 1 else 'slower'}")
        print(f"  Memory: {memory_reduction*100:.1f}% {'reduction' if memory_reduction > 0 else 'increase'}")
        print(f"  Parameters: {dynamic_params/granite_params:.2f}x")

        comparison['improvements'] = {
            'speed_ratio': speed_improvement,
            'memory_reduction_pct': memory_reduction * 100,
            'param_ratio': dynamic_params / granite_params
        }

    return comparison

def run_memory_scaling_comparison(
    dynamic_model: DynamicHybridModel,
    granite_model: Optional[GraniteNanoWrapper],
    config: ModelConfig,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Compare memory scaling across sequence lengths"""

    seq_lengths = [128, 256, 512, 1024, 2048]

    print("\n" + "="*70)
    print("MEMORY SCALING BENCHMARK")
    print("="*70)

    print("\nDynamicHybridModel Memory Scaling:")
    dynamic_scaling = benchmark_memory_scaling(
        dynamic_model,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size,
        device=device,
        seq_lengths=seq_lengths
    )

    results = {'dynamic': dynamic_scaling}

    if granite_model and granite_model.model is not None:
        print("\nGranite Nano Memory Scaling:")
        granite_scaling = benchmark_memory_scaling(
            granite_model,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
            device=device,
            seq_lengths=seq_lengths
        )
        results['granite'] = granite_scaling

        # Compare scaling rates
        print("\n" + "="*70)
        print("SCALING COMPARISON")
        print("="*70)
        print("\nSeq Length | Dynamic Mem | Granite Mem | Reduction")
        print("-" * 60)
        for seq_len in seq_lengths:
            if seq_len in dynamic_scaling and seq_len in granite_scaling:
                d_mem = dynamic_scaling[seq_len]['mean_memory_gb']
                g_mem = granite_scaling[seq_len]['mean_memory_gb']
                reduction = (1 - d_mem / g_mem) * 100
                print(f"{seq_len:10d} | {d_mem:11.2f} | {g_mem:11.2f} | {reduction:8.1f}%")

    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark DynamicHybridModel vs Granite Nano')
    parser.add_argument('--model', choices=['dynamic', 'granite', 'both'], default='both',
                       help='Which model to benchmark')
    parser.add_argument('--config', type=str, choices=list(CONFIG_PRESETS.keys()),
                       help=f'Use a predefined config preset: {", ".join(CONFIG_PRESETS.keys())}')
    parser.add_argument('--granite-path', type=str, default='~/models/granite-350m/',
                       help='Path to Granite Nano model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for benchmarking (overrides config preset)')
    parser.add_argument('--seq-length', type=int, default=None,
                       help='Sequence length for benchmarking (overrides config preset)')
    parser.add_argument('--memory-scaling', action='store_true',
                       help='Run memory scaling benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Configure DynamicHybridModel
    config = ModelConfig()

    # Apply config preset if specified
    if args.config:
        config = apply_config_preset(config, args.config)

    # Override with command-line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        print(f"   Overriding batch_size: {args.batch_size}")
    elif args.config is None:
        config.batch_size = 8  # Default if no config preset

    if args.seq_length is not None:
        config.max_seq_len = args.seq_length
        print(f"   Overriding max_seq_len: {args.seq_length}")
    elif args.config is None:
        config.max_seq_len = 1024  # Default if no config preset

    config.gradient_checkpointing = False  # Disable for benchmarking
    config.mixed_precision = (args.device == 'cuda')

    # Initialize models
    dynamic_model = None
    granite_model = None

    if args.model in ['dynamic', 'both']:
        print("Initializing DynamicHybridModel...")
        dynamic_model = DynamicHybridModel(config)
        dynamic_model.eval()
        if args.device == 'cuda' and torch.cuda.is_available():
            dynamic_model = dynamic_model.cuda()
        print(f"✓ DynamicHybridModel initialized ({sum(p.numel() for p in dynamic_model.parameters())/1e6:.1f}M params)")

    if args.model in ['granite', 'both']:
        granite_path = Path(args.granite_path).expanduser()
        if granite_path.exists():
            granite_model = GraniteNanoWrapper(str(granite_path), args.device)
        else:
            print(f"⚠ Granite model path not found: {granite_path}")
            print("  Skipping Granite benchmarks")

    # Run benchmarks
    all_results = {}

    if args.memory_scaling:
        scaling_results = run_memory_scaling_comparison(
            dynamic_model, granite_model, config, args.device
        )
        all_results['memory_scaling'] = scaling_results
    else:
        comparison_results = compare_models(
            dynamic_model, granite_model, config, args.device
        )
        all_results['comparison'] = comparison_results

    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        json.dump(convert_types(all_results), f, indent=2)

    print("✓ Benchmark complete!")

if __name__ == '__main__':
    main()
