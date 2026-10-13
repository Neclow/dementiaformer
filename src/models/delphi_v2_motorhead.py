"""Delphi v2 with a MOTOR head: hazards that change over time bins."""

from dataclasses import dataclass

from torch import Tensor, nn

from src.models._delphi import init_weights
from src.models.delphi_v2 import Backbone, DelphiV2Config


@dataclass
class DelphiV2MotorheadConfig(DelphiV2Config):
    """Settings of Delphi v2 with a MOTOR head: Delphi v2's and the time bins."""

    # Boundaries of the time bins, in years after a token. From age 70, 15 and
    # 25 years are the horizons 85 and 95
    bin_boundaries: tuple[float, ...] = (1, 2, 5, 10, 15, 25)


class DelphiV2Motorhead(nn.Module):
    """Delphi v2's backbone with a MOTOR head: one hazard per time bin and token."""

    def __init__(self, config: DelphiV2MotorheadConfig):
        super().__init__()
        self.config = config
        self.backbone = Backbone(config)
        self.n_bins = len(config.bin_boundaries) + 1
        # One vector per time bin, multiplied with the token embeddings
        self.bin_head = nn.Linear(
            config.n_embd, self.n_bins * config.n_embd, bias=False
        )
        init_weights(self, config.n_layer)

    def forward(
        self, tokens: Tensor, ages: Tensor, target_ages: Tensor | None = None
    ) -> Tensor:
        """Log hazards after each token: (batch, token, bin, vocabulary)."""
        hidden = self.backbone(tokens, ages, target_ages)
        bins = self.bin_head(hidden).unflatten(-1, (self.n_bins, -1))
        return bins @ self.backbone.wte.weight.T
