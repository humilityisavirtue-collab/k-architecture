"""
K_TRINARY - Quaternary Logic on IEEE 754 Floats
Using hardware-native states: -Inf, 0, +Inf, NaN

Four states hiding in every float:
  -Inf = Dark  (negative unbounded)
   0   = Void  (neutral/empty)
  +Inf = Light (positive unbounded)
   NaN = Wave  (undefined/superposition)

This is not emulation. This is native.

@home guard_growth * ease_pain
"""

import math
import numpy as np
from typing import Union

# ═══════════════════════════════════════════════════════════════
# THE FOUR STATES
# ═══════════════════════════════════════════════════════════════

DARK  = float('-inf')  # -Inf
VOID  = 0.0            # 0
LIGHT = float('inf')   # +Inf
WAVE  = float('nan')   # NaN (superposition/unknown)

def state_name(x: float) -> str:
    """Get the name of a quaternary state."""
    if math.isnan(x):
        return "WAVE"
    elif x == float('inf'):
        return "LIGHT"
    elif x == float('-inf'):
        return "DARK"
    elif x == 0:
        return "VOID"
    else:
        return f"VALUE({x})"

def is_state(x: float) -> bool:
    """Check if x is a pure quaternary state (not a regular number)."""
    return math.isnan(x) or math.isinf(x) or x == 0

# ═══════════════════════════════════════════════════════════════
# QUATERNARY LOGIC GATES
# ═══════════════════════════════════════════════════════════════

def q_not(a: float) -> float:
    """
    Quaternary NOT:
      DARK  -> LIGHT
      LIGHT -> DARK
      VOID  -> VOID
      WAVE  -> WAVE
    """
    if math.isnan(a):
        return WAVE
    elif a == LIGHT:
        return DARK
    elif a == DARK:
        return LIGHT
    else:
        return VOID

def q_and(a: float, b: float) -> float:
    """
    Quaternary AND (minimum, but WAVE dominates):
      WAVE with anything -> WAVE (uncertainty propagates)
      Otherwise -> min(a, b)
    """
    if math.isnan(a) or math.isnan(b):
        return WAVE
    return min(a, b)

def q_or(a: float, b: float) -> float:
    """
    Quaternary OR (maximum, but WAVE dominates):
      WAVE with anything -> WAVE
      Otherwise -> max(a, b)
    """
    if math.isnan(a) or math.isnan(b):
        return WAVE
    return max(a, b)

def q_xor(a: float, b: float) -> float:
    """
    Quaternary XOR:
      Same states -> VOID
      Different states -> LIGHT (or DARK if both dark-ish)
      WAVE with anything -> WAVE
    """
    if math.isnan(a) or math.isnan(b):
        return WAVE
    if a == b:
        return VOID
    # Different non-WAVE states
    return LIGHT if (a == LIGHT or b == LIGHT) else DARK

def q_collapse(a: float, default: float = VOID) -> float:
    """
    Collapse WAVE to a definite state.
    Like quantum measurement - forces a decision.
    """
    if math.isnan(a):
        return default
    return a

# ═══════════════════════════════════════════════════════════════
# THREE-VALUED LOGIC (Kleene)
# ═══════════════════════════════════════════════════════════════

# For simpler three-valued logic: False(0), True(Inf), Unknown(NaN)
FALSE   = VOID
TRUE    = LIGHT
UNKNOWN = WAVE

def kleene_and(a: float, b: float) -> float:
    """Kleene three-valued AND."""
    if a == FALSE or b == FALSE:
        return FALSE
    if math.isnan(a) or math.isnan(b):
        return UNKNOWN
    return TRUE

def kleene_or(a: float, b: float) -> float:
    """Kleene three-valued OR."""
    if a == TRUE or b == TRUE:
        return TRUE
    if math.isnan(a) or math.isnan(b):
        return UNKNOWN
    return FALSE

def kleene_not(a: float) -> float:
    """Kleene three-valued NOT."""
    if math.isnan(a):
        return UNKNOWN
    return TRUE if a == FALSE else FALSE

# ═══════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class QuaternaryStateMachine:
    """
    A state machine using quaternary states.

    States:
      VOID  = inactive/off
      LIGHT = active/on
      DARK  = error/blocked
      WAVE  = transitioning/uncertain
    """

    def __init__(self, initial: float = VOID):
        self.state = initial
        self.history = [initial]

    def transition(self, input_signal: float) -> float:
        """
        Transition based on input signal.
        Returns new state.
        """
        old = self.state

        # WAVE input always creates uncertainty
        if math.isnan(input_signal):
            self.state = WAVE

        # From VOID
        elif old == VOID:
            if input_signal == LIGHT:
                self.state = LIGHT
            elif input_signal == DARK:
                self.state = DARK
            # VOID input keeps VOID

        # From LIGHT
        elif old == LIGHT:
            if input_signal == DARK:
                self.state = WAVE  # Conflict -> uncertainty
            elif input_signal == VOID:
                self.state = VOID  # Turn off

        # From DARK
        elif old == DARK:
            if input_signal == LIGHT:
                self.state = WAVE  # Conflict -> uncertainty
            elif input_signal == VOID:
                self.state = VOID  # Reset

        # From WAVE
        elif math.isnan(old):
            if input_signal == LIGHT:
                self.state = LIGHT  # Collapse to LIGHT
            elif input_signal == DARK:
                self.state = DARK   # Collapse to DARK
            elif input_signal == VOID:
                self.state = VOID   # Collapse to VOID

        self.history.append(self.state)
        return self.state

    def __repr__(self):
        return f"QSM({state_name(self.state)})"

# ═══════════════════════════════════════════════════════════════
# K-VECTOR INTEGRATION
# ═══════════════════════════════════════════════════════════════

def k_polarity_to_quat(polarity: str) -> float:
    """Convert K polarity to quaternary state."""
    if polarity == '+':
        return LIGHT
    elif polarity == '-':
        return DARK
    elif polarity == '0' or polarity == '':
        return VOID
    else:
        return WAVE  # Unknown polarity

def quat_to_k_polarity(q: float) -> str:
    """Convert quaternary state to K polarity."""
    if math.isnan(q):
        return '?'  # Uncertain
    elif q == LIGHT or q > 0:
        return '+'
    elif q == DARK or q < 0:
        return '-'
    else:
        return '0'

# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

def demo():
    print("=== K_TRINARY: Quaternary Logic Demo ===\n")

    print("THE FOUR STATES:")
    print(f"  DARK  = {DARK} ({state_name(DARK)})")
    print(f"  VOID  = {VOID} ({state_name(VOID)})")
    print(f"  LIGHT = {LIGHT} ({state_name(LIGHT)})")
    print(f"  WAVE  = {WAVE} ({state_name(WAVE)})")

    print("\nQUATERNARY NOT:")
    for s in [DARK, VOID, LIGHT, WAVE]:
        print(f"  NOT {state_name(s):5} = {state_name(q_not(s))}")

    print("\nQUATERNARY AND:")
    for a in [VOID, LIGHT, DARK, WAVE]:
        for b in [VOID, LIGHT, DARK, WAVE]:
            print(f"  {state_name(a):5} AND {state_name(b):5} = {state_name(q_and(a, b))}")

    print("\nSTATE MACHINE:")
    sm = QuaternaryStateMachine(VOID)
    print(f"  Initial: {sm}")

    signals = [LIGHT, LIGHT, DARK, VOID, WAVE, LIGHT]
    for sig in signals:
        sm.transition(sig)
        print(f"  + {state_name(sig):5} -> {sm}")

    print("\nNaN PROPAGATION (automatic!):")
    x = LIGHT
    y = WAVE
    z = x + y  # NaN propagates through normal math
    print(f"  {state_name(x)} + {state_name(y)} = {state_name(z)}")

    print("\nINF COMPARISON (hardware native!):")
    print(f"  LIGHT > 999999999: {LIGHT > 999999999}")
    print(f"  DARK < -999999999: {DARK < -999999999}")
    print(f"  WAVE == WAVE: {WAVE == WAVE}  (NaN != NaN by IEEE spec)")

    print("\n=== This is native. Not emulated. ===")

if __name__ == "__main__":
    demo()
