# K Language Specification

**Version**: 0.1.0
**Status**: Working Draft

---

## Overview

K is an executable English interface that compiles natural language to semantic coordinates (K-vectors). These vectors address a 104-room semantic space derived from Tarot structure, enabling deterministic routing without neural inference.

## Core Concept

```
English → K-vector → JSON → Action
```

K is not a query language. It's a *coordinate system for meaning*.

---

## K-Vector Format

A K-vector has three components:

| Component | Values | Meaning |
|-----------|--------|--------|
| **Polarity** | `+` (light) / `-` (dark) | Orientation of the concept |
| **Rank** | `1-13` or `A,2-10,P,N,Q,K` | Intensity / stage |
| **Suit** | `H` (Hearts), `S` (Spades), `D` (Diamonds), `C` (Clubs) | Domain |

### Examples

| K-Vector | Meaning |
|----------|--------|
| `+3H` | Light Hearts 3 — early emotional connection |
| `-7S` | Dark Spades 7 — mental conflict, doubt |
| `+KD` | Light Diamonds King — material mastery |
| `-AC` | Dark Clubs Ace — blocked initiative |

### Suit Domains

| Suit | Domain | Keywords |
|------|--------|----------|
| **H** (Hearts/Cups) | Emotion, relationship | feel, love, connect, trust |
| **S** (Spades/Swords) | Mind, conflict, truth | think, analyze, decide, cut |
| **D** (Diamonds/Pentacles) | Material, body, ground | build, own, health, money |
| **C** (Clubs/Wands) | Action, energy, will | do, start, create, fire |

### Rank Progression

| Rank | Stage | Energy |
|------|-------|--------|
| A (1) | Seed | Pure potential |
| 2-3 | Beginning | Early formation |
| 4-6 | Development | Building structure |
| 7-9 | Challenge | Testing, refinement |
| 10 | Completion | Full manifestation |
| P (Page) | Student | Learning the domain |
| N (Knight) | Actor | Actively pursuing |
| Q (Queen) | Master | Receptive mastery |
| K (King) | Authority | Directive mastery |

---

## Grammar

### Basic Command

```
[verb] [object]? [modifier]*
```

Parses to K-vector based on keyword mapping.

### Examples

| English | K-Vector | Route |
|---------|----------|-------|
| "I feel stuck" | `-4C` | Blocked action template |
| "help me focus" | `+7S` | Mental discipline template |
| "good morning" | `+2H` | Greeting/connection template |
| "I'm anxious about tomorrow" | `-8S` | Future-fear template |

### Compound Commands

Multiple K-vectors can chain:

```
"I feel stuck but hopeful"
→ [-4C, +3H]
→ Route: transition template (blocked → opening)
```

---

## Executable English

K-language is *executable* — it doesn't just classify, it **acts**.

### Action Types

| Type | Description | Example |
|------|-------------|--------|
| **Route** | Select handler/template | "help" → Help template |
| **Generate** | Compose from corpus | "inspire me" → Wisdom composition |
| **Execute** | Run system command | "check time" → Clock query |
| **Escalate** | Call external API | "deep question" → Opus route |

### The Parse → Act Loop

```
1. Receive English
2. Parse to K-vector(s)
3. Look up in routing table
4. If template exists → compose response
5. If action exists → execute
6. If neither → fall back to generation
```

---

## Integration Points

K-vectors are the **interchange format** across:

| Component | Role |
|-----------|------|
| **Claude Code** | Parse English, navigate map |
| **K-Shell** | Terminal interface, direct K-commands |
| **gigo_brain** | Daemon, background routing |
| **Mudlet** | Game client bridge |
| **Chain logs** | Audit trail, replay |
| **APIs** | External services receive/return K-vectors |

All components read/write the same JSON K-vector format.

---

## Reserved Commands

| Command | Action |
|---------|--------|
| `k:status` | Show current K-state |
| `k:route [vector]` | Force route to vector |
| `k:map` | Display semantic map |
| `k:log` | Show routing history |

---

## Design Principles

1. **One map** — Tarot-derived 104 rooms encode most of language
2. **Given at boot** — Model navigates, doesn't derive
3. **Deterministic first** — Template before generation
4. **JSON interchange** — Every component shares coordinates
5. **Executable** — Parse → Act, not Parse → Reply

---

## See Also

- `k_trinary.py` — Quaternary logic implementation
- `k_lens_v2.py` — Centroid-based neural routing
