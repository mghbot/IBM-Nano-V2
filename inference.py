"""
Inference script for DynamicHybridModel.

Usage:
    # Generate text
    python inference.py --checkpoint model.pt --prompt "Hello world" --max-length 100

    # Interactive mode
    python inference.py --checkpoint model.pt --interactive

    # Batch inference
    python inference.py --checkpoint model.pt --input prompts.txt --output results.txt
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from typing import List, Optional
import json
import time

from model import DynamicHybridModel, ModelConfig

# =============================================================================
# TEXT GENERATION
# =============================================================================

class SimpleTokenizer:
    """Simple character-level tokenizer for demo purposes"""

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        # Simple character-level encoding
        tokens = []
        for char in text:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                if idx >= self.vocab_size:
                    idx = self.vocab_size - 1  # UNK token
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
            tokens.append(self.char_to_idx[char])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char.get(t, '�') for t in tokens])

# Try to use real tokenizer if available
try:
    from transformers import AutoTokenizer

    def create_tokenizer(model_path: Optional[str] = None):
        """Create tokenizer from model or use GPT-2 tokenizer"""
        if model_path and Path(model_path).exists():
            return AutoTokenizer.from_pretrained(model_path)
        else:
            # Fallback to GPT-2 tokenizer
            try:
                return AutoTokenizer.from_pretrained('gpt2')
            except:
                return SimpleTokenizer()
except ImportError:
    def create_tokenizer(model_path: Optional[str] = None):
        return SimpleTokenizer()

# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """Inference engine for DynamicHybridModel"""

    def __init__(
        self,
        model: DynamicHybridModel,
        tokenizer,
        device: str = 'cuda',
        use_amp: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and device == 'cuda'

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        else:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)])

        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]

        # Replicate for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)

        # Track generated tokens for repetition penalty
        generated_tokens = input_ids.clone()

        # Generate
        for _ in range(max_length):
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits, _ = self.model(input_ids)

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_tokens[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

            # Check if all sequences have generated EOS (assuming EOS = 0)
            # if (next_token == 0).all():
            #     break

        # Decode
        outputs = []
        for i in range(num_return_sequences):
            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(input_ids[i].tolist())
            outputs.append(text)

        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        outputs = []
        for prompt in prompts:
            generated = self.generate(prompt, num_return_sequences=1, **kwargs)
            outputs.extend(generated)
        return outputs

# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load model checkpoint"""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config and state dict
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to infer config
            config = ModelConfig()

        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
    else:
        config = ModelConfig()
        state_dict = checkpoint

    # Create model
    model = DynamicHybridModel(config)
    model.load_state_dict(state_dict)

    print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    return model, config

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(engine: InferenceEngine, **gen_kwargs):
    """Interactive text generation mode"""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
    print("Type 'config' to see/change generation parameters.")
    print("="*70 + "\n")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'config':
                print("\nCurrent generation config:")
                for key, value in gen_kwargs.items():
                    print(f"  {key}: {value}")
                continue

            if not prompt:
                continue

            # Generate
            print("\nGenerating...")
            start = time.time()
            outputs = engine.generate(prompt, **gen_kwargs)
            elapsed = time.time() - start

            # Display results
            for i, output in enumerate(outputs):
                print(f"\n--- Output {i+1} ---")
                print(output)

            print(f"\n(Generated in {elapsed:.2f}s)")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")

# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_process(
    engine: InferenceEngine,
    input_file: str,
    output_file: str,
    **gen_kwargs
):
    """Process prompts from file"""
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read prompts
    with open(input_path) as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(prompts)} prompts...")

    # Generate
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
        outputs = engine.generate(prompt, num_return_sequences=1, **gen_kwargs)
        results.append({
            'prompt': prompt,
            'output': outputs[0]
        })

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DynamicHybridModel Inference')

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                       help='Path to tokenizer (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')

    # Mode args
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt to generate from')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file with prompts (one per line)')
    parser.add_argument('--output', type=str, default='outputs.json',
                       help='Output file for batch processing')

    # Generation args
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling threshold')
    parser.add_argument('--repetition-penalty', type=float, default=1.0,
                       help='Repetition penalty')
    parser.add_argument('--num-return-sequences', type=int, default=1,
                       help='Number of sequences to generate')

    args = parser.parse_args()

    # Load model
    model, config = load_checkpoint(args.checkpoint)

    # Create tokenizer
    tokenizer = create_tokenizer(args.tokenizer_path)

    # Create engine
    engine = InferenceEngine(
        model,
        tokenizer,
        device=args.device,
        use_amp=(args.device == 'cuda')
    )

    # Generation kwargs
    gen_kwargs = {
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'num_return_sequences': args.num_return_sequences
    }

    # Run inference
    if args.interactive:
        interactive_mode(engine, **gen_kwargs)

    elif args.input:
        batch_process(engine, args.input, args.output, **gen_kwargs)

    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        print("Generating...")
        start = time.time()
        outputs = engine.generate(args.prompt, **gen_kwargs)
        elapsed = time.time() - start

        for i, output in enumerate(outputs):
            print(f"\n--- Output {i+1} ---")
            print(output)

        print(f"\n(Generated in {elapsed:.2f}s)")

    else:
        print("Please specify --interactive, --prompt, or --input")
        parser.print_help()

if __name__ == '__main__':
    main()
