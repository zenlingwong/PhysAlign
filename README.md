# PhysAlign

PhysAlign is a physics-guided parameter-efficient fine-tuning method that coordinates low-rank capacity, observed physical-response supervision, and the optimizer's realized update.

## Method

1. `model/projector.py` calibrates an uncentered activation second moment, augments the visible basis with the activation mean when needed, completes the orthogonal complement, and allocates a fixed rank budget using projected gradient spectra.
2. `model/routed_lora.py` implements the routed update
   `Delta W = alpha / r * (Bv Cv Qv.T + Bc Cc Qc.T)`.
3. `src/horizontal_momentum.py` provides the centered spherical-grid response operator used for flow supervision.
4. `src/losses.py` matches the predicted response to the detached observed response, so the observed state is a common target for state and physics losses.
5. `src/update.py` projects the actual AdamW displacement onto the joint first-order feasible cone and writes the corrected displacement back to the trainable factors.

## Layout

```text
PhysAlign/
├── README.md
├── config/
│   └── default.json
├── model/
│   ├── __init__.py
│   ├── projector.py
│   └── routed_lora.py
└── src/
    ├── __init__.py
    ├── horizontal_momentum.py
    ├── losses.py
    └── update.py
```

## Requirements

- Python 3.10 or newer
- PyTorch

## Minimal usage

```python
import torch

from PhysAlign.model import ProjectorBank, ProjectorConfig, RoutedLoRAConfig, RoutedLoRALinear
from PhysAlign.src import corrected_optimizer_step, normalized_state_loss, observed_response_loss

calibration = torch.randn(32, 64)
projector = ProjectorBank.calibrate(calibration, ProjectorConfig(energy_threshold=0.999))
visible_rank, complement_rank = projector.allocate_rank(torch.randn(128, 64), total_rank=8)
adapter = RoutedLoRALinear(
    torch.randn(128, 64),
    projector,
    RoutedLoRAConfig(visible_rank, complement_rank, alpha=8.0),
)
```

The training loop should compute the data and observed-response gradients at one shared prediction and pass the trainable parameters and both flattened gradients to `corrected_optimizer_step`, which performs the optimizer step and corrects its realized displacement.
