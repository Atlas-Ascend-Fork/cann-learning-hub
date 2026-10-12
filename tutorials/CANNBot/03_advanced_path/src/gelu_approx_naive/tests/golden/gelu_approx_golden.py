"""gelu_approx golden reference (numpy float64)."""
import numpy as np

def gelu_approx_golden(x_f32):
    """Exact GELU approximation in float64, rounded to float32."""
    x = x_f32.astype(np.float64)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    z = 0.5 * x * (1.0 + np.tanh(inner))
    return z.astype(np.float32)
