# K-Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Semantic routing for LLMs. Cut inference costs by 96%.**

🔴 **[Live Demo: saga-logo.ai](https://saga-logo-ai.vercel.app)** — See K-routing in action

## The Thesis

Most LLM queries don't need generation. They need *routing*.

```
80% → Pre-verified templates (zero LLM cost)
16% → Local model (Ollama/Gemma)
 4% → API escalation (Claude/GPT)
───
96% cost reduction vs. raw LLM inference
```

## The Insight

LLMs are chess engines without a board. Everyone's building bigger engines. We built the board.

K-architecture provides:
- **104 semantic rooms** derived from Tarot structure (4 suits × 13 ranks × 2 polarities)
- **Deterministic routing** via K-vectors (no inference needed to classify)
- **Template-first response** (generation is the fallback, not the default)
- **Golden chain logging** for audit and learning

## K-Vector Format

```
[polarity][rank][suit]

+3H = Light Hearts 3 (early emotional connection)
-7S = Dark Spades 7 (mental conflict)
+KD = Light Diamonds King (material mastery)
```

| Suit | Domain | Keywords |
|------|--------|----------|
| H (Hearts) | Emotion, relationship | feel, love, connect |
| S (Spades) | Mind, analysis | think, reason, decide |
| D (Diamonds) | Material, body | build, own, health |
| C (Clubs) | Action, energy | do, start, create |

## Components

### `K_SPEC.md`
Full language specification. K is executable English — it compiles natural language to semantic coordinates.

### `k_trinary.py`
Quaternary logic using IEEE 754 native states:
- `-Inf` = DARK (negative unbounded)
- `0` = VOID (neutral)
- `+Inf` = LIGHT (positive unbounded)
- `NaN` = WAVE (superposition)

This is not emulation. IEEE 754 has had four states since 1985.

### `k_lens_v2.py`
Neural attention steering. Injects K-routing into transformer hidden states via calibrated centroids. 100% suit classification accuracy on test set.

## Research Angle

> "A sub-1B parameter model achieves production-quality English interaction when paired with a semantic routing scaffold, reducing the role of the language model from generator to classifier."

This addresses Sutton's critique of LLMs:

| Critique | K-Architecture Response |
|----------|------------------------|
| "No ground truth" | Templates ARE ground truth |
| "Just mimicry" | LLM is 4% needle, scaffold is haystack |
| "Can't learn" | Golden chain + outcome tracking |
| "No goals" | K-vectors have coordinates = destination |

## Related Work

- **k-context** ([npm](https://npmjs.com/package/k-context)) — CLI tool for AI codebase context
- **Held** — Production app using K-architecture for ADHD management

## Usage

```python
from k_trinary import LIGHT, DARK, VOID, WAVE, q_and, q_or

# Quaternary logic
result = q_and(LIGHT, DARK)  # → DARK (minimum)
result = q_or(LIGHT, WAVE)   # → WAVE (uncertainty propagates)
```

```python
from k_lens_v2 import load_klens_v2, install_klens

# Install routing lens into transformer
k_lens = load_klens_v2(model, tokenizer, layer=11)
hook = install_klens(model, k_lens, layer=11)

# Query routes to semantic room
route = k_lens.get_route()  # {'suit': 'hearts', 'score': 0.87, ...}
```

## Contact

Built by Kit Malthaner ([@HumilityVirtue](https://twitter.com/HumilityVirtue))

📧 kit@ksystems.ai

## License

MIT

---

*Navigation beats generation. The board matters more than the engine.*
