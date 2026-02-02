"""
K-LENS V2: Centroid-based routing with 52 card rooms

Uses hidden state centroids measured from TinyLlama layer 11.
"""

import torch
import torch.nn as nn
import math
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + math.sqrt(5)) / 2
DEFAULT_AGGRESSION = 1 / (PHI ** 2)

# Calibration prompts for each suit (8 per suit = 32 total)
CALIBRATION = {
    "hearts": [
        "I love you", "feeling sad", "emotional warmth", "my heart aches",
        "compassion", "grief and tears", "joy and happiness", "loving feeling"
    ],
    "spades": [
        "calculate this", "logical analysis", "think carefully", "reasoning",
        "deduce the answer", "solve equation", "proof theorem", "analyze data"
    ],
    "diamonds": [
        "how much money", "gold coins", "price and cost", "wealthy rich",
        "buy this item", "expensive purchase", "dollar value", "sell for profit"
    ],
    "clubs": [
        "build it now", "take action", "move forward", "create something",
        "run fast", "strike hard", "make this", "do it immediately"
    ],
}

SUITS = ["hearts", "spades", "diamonds", "clubs"]
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def build_room_names():
    """Build all 52 card room names + origin."""
    names = []
    for suit in SUITS:
        symbol = suit[0].upper()
        for rank in RANKS:
            names.append(f"{rank}{symbol}")
    names.append("Origin")
    return names


class KLensV2(nn.Module):
    """K-lens with centroid-based suit routing."""

    def __init__(self, centroids, aggression=DEFAULT_AGGRESSION):
        super().__init__()
        self.aggression = aggression
        self.room_names = build_room_names()
        self.last_route = None

        # Store suit centroids
        # centroids: dict of suit -> tensor [hidden_dim]
        for suit, centroid in centroids.items():
            self.register_buffer(f'centroid_{suit}', centroid)

        self.centroids = centroids

    def forward(self, x):
        """Apply K-lens transformation with centroid routing."""
        # x: [batch, seq, hidden_dim]

        # Average hidden states (skip first token)
        if x.size(1) > 1:
            content_avg = x[0, 1:, :].mean(dim=0)
        else:
            content_avg = x[0, 0, :]

        # Compute cosine similarity to each suit centroid
        similarities = {}
        for suit in SUITS:
            centroid = getattr(self, f'centroid_{suit}')
            sim = torch.cosine_similarity(content_avg.unsqueeze(0), centroid.unsqueeze(0))
            similarities[suit] = sim.item()

        # Find best suit
        best_suit = max(similarities, key=similarities.get)
        best_score = similarities[best_suit]

        # For now, assign to Ace of that suit (rank 0)
        suit_idx = SUITS.index(best_suit)
        room_idx = suit_idx * 13  # Ace of suit
        room_name = self.room_names[room_idx]

        self.last_route = {
            'room': room_name,
            'suit': best_suit,
            'score': best_score,
            'similarities': similarities
        }

        # Soft routing: blend toward suit centroid
        target_centroid = getattr(self, f'centroid_{best_suit}')

        # Per-token blending toward centroid direction
        # This nudges all hidden states toward the suit's semantic direction
        centroid_expanded = target_centroid.unsqueeze(0).unsqueeze(0).expand_as(x)
        out = x + self.aggression * (centroid_expanded - x) * 0.1  # Gentle nudge

        return out

    def get_route(self):
        return self.last_route

    def get_suit(self):
        return self.last_route['suit'] if self.last_route else None


def calibrate_from_model(model, tokenizer, layer=11):
    """Compute hidden state centroids for each suit."""
    print(f"  Calibrating centroids from layer {layer}...")

    centroids = {}
    hidden_states_by_suit = {suit: [] for suit in SUITS}

    # Collect hidden states for each calibration prompt
    for suit, prompts in CALIBRATION.items():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(inputs.input_ids, output_hidden_states=True)
                # Get hidden state from target layer
                hidden = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
                # Average over tokens (skip first)
                if hidden.size(1) > 1:
                    avg = hidden[0, 1:, :].mean(dim=0)
                else:
                    avg = hidden[0, 0, :]
                hidden_states_by_suit[suit].append(avg)

    # Compute centroids
    for suit in SUITS:
        states = torch.stack(hidden_states_by_suit[suit])
        centroids[suit] = states.mean(dim=0)
        print(f"    {suit}: {len(hidden_states_by_suit[suit])} samples")

    return centroids


def load_klens_v2(model, tokenizer, layer=11):
    """Create a calibrated K-lens V2."""
    centroids = calibrate_from_model(model, tokenizer, layer)
    return KLensV2(centroids)


def install_klens(model, k_lens, layer=11):
    """Install K-lens into transformer at specified layer."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = k_lens(hidden)
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    return model.model.layers[layer].register_forward_hook(hook)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  K-LENS V2: Centroid-based routing")
    print("="*60)

    # Load model
    print("\n  Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32
    )
    model.eval()

    # Create and install K-lens
    print("\n  Creating K-lens V2...")
    k_lens = load_klens_v2(model, tokenizer, layer=11)
    hook = install_klens(model, k_lens, layer=11)

    print(f"\n  Rooms: {len(k_lens.room_names)}")
    print(f"  Aggression: {k_lens.aggression:.4f}")

    # Test routing
    print("\n" + "-"*60)
    print("  ROUTING TESTS")
    print("-"*60)

    test_prompts = [
        ("I love you so much", "hearts"),
        ("My heart is broken", "hearts"),
        ("Feeling emotional", "hearts"),

        ("Calculate 5+5", "spades"),
        ("Analyze the data", "spades"),
        ("Think logically", "spades"),

        ("How much does it cost?", "diamonds"),
        ("Count the gold", "diamonds"),
        ("What is the price?", "diamonds"),

        ("Build it now!", "clubs"),
        ("Take action", "clubs"),
        ("Create something", "clubs"),
    ]

    correct = 0
    for prompt, expected in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)

        route = k_lens.get_route()
        if route:
            actual = route['suit']
            match = "OK" if actual == expected else "xx"
            if actual == expected:
                correct += 1
            print(f"  {match} {prompt:<25} {expected:<10} {actual:<10} {route['score']:.3f}")

    print("-"*60)
    pct = 100 * correct / len(test_prompts)
    print(f"  ACCURACY: {correct}/{len(test_prompts)} ({pct:.0f}%)")

    hook.remove()
    print("\n  Done. Dai stiho.")
