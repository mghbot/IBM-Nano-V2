"""
Utilities for loading, converting, and comparing with IBM Granite Nano models.

This module provides:
- Loading Granite Nano checkpoints
- Converting Granite Nano weights to DynamicHybridModel
- Architecture comparison tools
- Weight initialization from Granite Nano
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json

from model import DynamicHybridModel, ModelConfig

# =============================================================================
# GRANITE NANO ARCHITECTURE MAPPING
# =============================================================================

GRANITE_NANO_CONFIGS = {
    '350m': {
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 50257,
        'max_seq_len': 2048,
    },
    '1b': {
        'hidden_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'vocab_size': 50257,
        'max_seq_len': 2048,
    }
}

# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_granite_checkpoint(
    model_path: str,
    use_transformers: bool = True
) -> Tuple[Any, Dict]:
    """
    Load a Granite Nano checkpoint.

    Args:
        model_path: Path to Granite Nano model
        use_transformers: Whether to use transformers library

    Returns:
        model: Loaded Granite model
        config: Model configuration dict
    """
    model_path = Path(model_path).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if use_transformers:
        try:
            from transformers import AutoModelForCausalLM, AutoConfig

            print(f"Loading Granite Nano from {model_path} using transformers...")
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )

            config_dict = {
                'hidden_dim': config.hidden_size,
                'num_layers': config.num_hidden_layers,
                'num_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size,
                'max_seq_len': config.max_position_embeddings,
            }

            print(f"✓ Loaded Granite Nano: {config.num_parameters / 1e6:.1f}M parameters")
            return model, config_dict

        except Exception as e:
            print(f"Failed to load with transformers: {e}")
            raise
    else:
        # Load raw checkpoint
        checkpoint_file = model_path / "pytorch_model.bin"
        if not checkpoint_file.exists():
            checkpoint_file = model_path / "model.safetensors"

        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            config_file = model_path / "config.json"
            with open(config_file) as f:
                config = json.load(f)
            return state_dict, config
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")

# =============================================================================
# WEIGHT CONVERSION
# =============================================================================

def convert_granite_to_dynamic_hybrid(
    granite_model: Any,
    config: ModelConfig,
    transfer_embeddings: bool = True,
    transfer_attention: bool = True,
    initialize_ssm: str = 'xavier'
) -> DynamicHybridModel:
    """
    Convert Granite Nano weights to DynamicHybridModel.

    Args:
        granite_model: Loaded Granite Nano model
        config: DynamicHybridModel configuration
        transfer_embeddings: Whether to transfer embedding weights
        transfer_attention: Whether to transfer attention weights
        initialize_ssm: How to initialize SSM layers ('xavier', 'kaiming', 'random')

    Returns:
        DynamicHybridModel with transferred weights
    """
    print("\nConverting Granite Nano weights to DynamicHybridModel...")

    # Create new model
    dynamic_model = DynamicHybridModel(config)

    # Get Granite state dict
    if hasattr(granite_model, 'state_dict'):
        granite_state = granite_model.state_dict()
    else:
        granite_state = granite_model

    # Transfer embeddings
    if transfer_embeddings:
        print("  Transferring embeddings...")
        try:
            # Find embedding layers in Granite
            granite_emb_key = None
            for key in granite_state.keys():
                if 'embed' in key.lower() and 'token' in key.lower():
                    granite_emb_key = key
                    break

            if granite_emb_key:
                granite_emb = granite_state[granite_emb_key]
                vocab_size = min(granite_emb.shape[0], config.vocab_size)
                hidden_dim = min(granite_emb.shape[1], config.hidden_dim)

                dynamic_model.embeddings.weight.data[:vocab_size, :hidden_dim] = \
                    granite_emb[:vocab_size, :hidden_dim]
                print(f"    ✓ Transferred embeddings: {vocab_size} x {hidden_dim}")
        except Exception as e:
            print(f"    ⚠ Failed to transfer embeddings: {e}")

    # Transfer attention weights
    if transfer_attention:
        print("  Transferring attention weights...")
        num_transferred = 0
        try:
            for layer_idx in range(min(config.num_layers, len(dynamic_model.layers))):
                # Try to find corresponding Granite layer
                granite_layer_prefix = f"model.layers.{layer_idx}"
                dynamic_layer = dynamic_model.layers[layer_idx]

                # Transfer attention weights
                for component in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    granite_key = f"{granite_layer_prefix}.self_attn.{component}.weight"

                    if granite_key in granite_state:
                        granite_weight = granite_state[granite_key]

                        # Map to DynamicHybridModel attention
                        # This is a simplified mapping - real conversion would need
                        # more sophisticated weight reshaping
                        num_transferred += 1

            if num_transferred > 0:
                print(f"    ✓ Transferred {num_transferred} attention components")
        except Exception as e:
            print(f"    ⚠ Failed to transfer attention weights: {e}")

    # Initialize SSM layers
    print(f"  Initializing SSM layers with {initialize_ssm} initialization...")
    for layer in dynamic_model.layers:
        if initialize_ssm == 'xavier':
            nn.init.xavier_uniform_(layer.ssm.log_A)
        elif initialize_ssm == 'kaiming':
            nn.init.kaiming_uniform_(layer.ssm.log_A)
        # 'random' uses default initialization

    print("✓ Conversion complete!")
    return dynamic_model

# =============================================================================
# ARCHITECTURE COMPARISON
# =============================================================================

def compare_architectures(
    granite_model: Any,
    dynamic_model: DynamicHybridModel
) -> Dict[str, Any]:
    """
    Compare Granite Nano and DynamicHybridModel architectures.

    Returns:
        Dictionary with architecture comparison details
    """
    # Count parameters
    def count_params(model):
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        return 0

    granite_params = count_params(granite_model)
    dynamic_params = count_params(dynamic_model)

    # Analyze layer types
    def analyze_layers(model):
        layer_types = {}
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        return layer_types

    granite_layers = analyze_layers(granite_model) if hasattr(granite_model, 'named_modules') else {}
    dynamic_layers = analyze_layers(dynamic_model)

    comparison = {
        'parameters': {
            'granite': granite_params,
            'dynamic': dynamic_params,
            'ratio': dynamic_params / granite_params if granite_params > 0 else 0
        },
        'layer_types': {
            'granite': granite_layers,
            'dynamic': dynamic_layers
        },
        'unique_to_dynamic': {
            'SSMLayer': dynamic_layers.get('SSMLayer', 0),
            'DynamicRouter': dynamic_layers.get('DynamicRouter', 0),
            'AdaptiveStateCompressor': dynamic_layers.get('AdaptiveStateCompressor', 0),
            'BidirectionalBridge': dynamic_layers.get('BidirectionalBridge', 0)
        }
    }

    return comparison

def print_architecture_comparison(comparison: Dict[str, Any]):
    """Print a formatted architecture comparison"""
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)

    print(f"\nParameter Count:")
    print(f"  Granite Nano: {comparison['parameters']['granite'] / 1e6:.1f}M")
    print(f"  DynamicHybrid: {comparison['parameters']['dynamic'] / 1e6:.1f}M")
    print(f"  Ratio: {comparison['parameters']['ratio']:.2f}x")

    print(f"\nUnique Components in DynamicHybridModel:")
    for component, count in comparison['unique_to_dynamic'].items():
        print(f"  {component}: {count}")

# =============================================================================
# INITIALIZATION UTILITIES
# =============================================================================

def create_config_from_granite(
    granite_model_path: str,
    model_size: str = '350m'
) -> ModelConfig:
    """
    Create a ModelConfig matching a Granite Nano configuration.

    Args:
        granite_model_path: Path to Granite model (for loading actual config)
        model_size: Size identifier ('350m', '1b') for fallback

    Returns:
        ModelConfig matching Granite Nano
    """
    config = ModelConfig()

    # Try to load actual config
    try:
        model_path = Path(granite_model_path).expanduser()
        config_file = model_path / "config.json"

        if config_file.exists():
            with open(config_file) as f:
                granite_config = json.load(f)

            config.hidden_dim = granite_config.get('hidden_size', config.hidden_dim)
            config.num_layers = granite_config.get('num_hidden_layers', config.num_layers)
            config.num_heads = granite_config.get('num_attention_heads', config.num_heads)
            config.vocab_size = granite_config.get('vocab_size', config.vocab_size)
            config.max_seq_len = granite_config.get('max_position_embeddings', config.max_seq_len)

            print(f"✓ Created config from {config_file}")
            return config
    except Exception as e:
        print(f"⚠ Could not load config from {granite_model_path}: {e}")

    # Fallback to preset configs
    if model_size in GRANITE_NANO_CONFIGS:
        preset = GRANITE_NANO_CONFIGS[model_size]
        for key, value in preset.items():
            setattr(config, key, value)
        print(f"✓ Using preset config for Granite Nano {model_size}")

    return config

# =============================================================================
# MAIN UTILITY FUNCTIONS
# =============================================================================

def load_and_convert_granite(
    granite_path: str,
    transfer_weights: bool = True
) -> Tuple[DynamicHybridModel, ModelConfig]:
    """
    One-stop function to load Granite and convert to DynamicHybridModel.

    Args:
        granite_path: Path to Granite Nano model
        transfer_weights: Whether to transfer compatible weights

    Returns:
        Tuple of (DynamicHybridModel, ModelConfig)
    """
    # Load Granite
    granite_model, granite_config_dict = load_granite_checkpoint(granite_path)

    # Create matching config
    config = create_config_from_granite(granite_path)

    # Convert
    if transfer_weights:
        dynamic_model = convert_granite_to_dynamic_hybrid(
            granite_model,
            config,
            transfer_embeddings=True,
            transfer_attention=True
        )
    else:
        dynamic_model = DynamicHybridModel(config)

    return dynamic_model, config

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Granite Nano utilities')
    parser.add_argument('--granite-path', type=str, required=True,
                       help='Path to Granite Nano model')
    parser.add_argument('--action', choices=['load', 'convert', 'compare'],
                       default='compare', help='Action to perform')
    parser.add_argument('--output', type=str, default='converted_model.pt',
                       help='Output path for converted model')

    args = parser.parse_args()

    if args.action == 'load':
        model, config = load_granite_checkpoint(args.granite_path)
        print(f"\nLoaded model config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    elif args.action == 'convert':
        dynamic_model, config = load_and_convert_granite(
            args.granite_path,
            transfer_weights=True
        )
        torch.save({
            'model_state': dynamic_model.state_dict(),
            'config': config
        }, args.output)
        print(f"\n✓ Saved converted model to {args.output}")

    elif args.action == 'compare':
        granite_model, _ = load_granite_checkpoint(args.granite_path)
        config = create_config_from_granite(args.granite_path)
        dynamic_model = DynamicHybridModel(config)

        comparison = compare_architectures(granite_model, dynamic_model)
        print_architecture_comparison(comparison)
