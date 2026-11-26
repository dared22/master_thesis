"""Zero-shot Time Series Foundation Model (TSFM) portfolio risk benchmark.

Forecasts a 1D daily-rebalanced portfolio series (cumulative log-wealth or
daily returns) strictly causally and computes empirical VaR and ES from
generated sample trajectories using the same conventions as historical
simulation. Deep-learning dependencies are imported lazily only when a
concrete neural forecaster is invoked.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import platform
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ._common import (
    ES_INTEGER_TOLERANCE,
    WEIGHT_TOLERANCE,
    ReturnInput,
    _coverage_table,
    _dated_table,
    _empirical_es,
    _empirical_var,
    _load,
    _portfolio_returns,
    _resolve_weights,
    _tail_mass,
    _validate_assets,
    _validate_choice,
    _validate_confidence_levels,
    _validate_int,
    _validate_weights,
)

SUNDIAL_MAX_CONTEXT = 2880
SUNDIAL_DEFAULT_MODEL_ID = 'thuml/sundial-base-128m'
TIMESTRAL_MAX_CONTEXT = 32768
TIMESTRAL_DEFAULT_MODEL_ID = 'mistralai/Timestral-28M'
TIMESTRAL_QUANTILE_LEVELS = (
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
)
DEFAULT_MODEL_IDS = {
    'sundial': SUNDIAL_DEFAULT_MODEL_ID,
    'timestral': TIMESTRAL_DEFAULT_MODEL_ID,
}
QUANTILE_LEVEL_TOLERANCE = 1e-9
QUANTILE_CROSSING_TOLERANCE = 1e-12


class SampleForecaster(Protocol):
    """Minimal interface for univariate probabilistic sample-path forecasters."""

    name: str

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable provenance for the forecaster."""
        ...

    def sample(
        self,
        contexts: np.ndarray,
        horizon: int,
        num_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Generate sample paths of shape (batch_size, num_samples, horizon)."""
        ...


class QuantileForecaster(Protocol):
    """Minimal interface for univariate quantile-head probabilistic forecasters."""

    name: str

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable provenance for the forecaster."""
        ...

    def predict_quantiles(
        self,
        contexts: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Return quantiles of shape (batch_size, n_levels, horizon) and increasing levels."""
        ...


Forecaster = SampleForecaster | QuantileForecaster


class SundialForecaster:
    """Lazy-loading wrapper around THUML's Sundial flow-matching TSFM."""

    name = 'sundial'

    def __init__(
        self,
        model_id: str = SUNDIAL_DEFAULT_MODEL_ID,
        device: str = 'cpu',
        dtype: str = 'float32',
        sample_chunk_size: int | None = None,
        revin: bool = True,
        revision: str | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError('model_id must be a nonempty string')
        _validate_choice('device', device, ('cpu', 'mps', 'cuda', 'auto'))
        _validate_choice('dtype', dtype, ('float32', 'float16', 'bfloat16'))
        sample_chunk_size = _validate_int('sample_chunk_size', sample_chunk_size, 1, allow_none=True)
        if not isinstance(revin, bool):
            raise ValueError('revin must be a bool')
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError('revision must be None or a nonempty string')
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.sample_chunk_size = sample_chunk_size
        self.revin = revin
        self.revision = revision
        self._model: Any = None
        self._resolved_device: str | None = None
        self._versions: dict[str, str] = {}

    def _load_model(self) -> tuple[Any, Any, str, Any]:
        try:
            import torch
            import transformers
        except ImportError as error:
            raise ImportError(
                'SundialForecaster requires torch and transformers==4.40.1. '
                'Install the optional extra with `uv sync --extra tsfm` (or '
                'run with `uv run --extra tsfm`).'
            ) from error
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        torch_dtype = dtype_map[self.dtype]
        if self._model is None:
            if self.device == 'auto':
                # Avoid auto-selecting Apple MPS: Sundial's flow-matching
                # sampling produces nonfinite values on PyTorch MPS (M2).
                resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                if self.device == 'mps' and not torch.backends.mps.is_available():
                    raise RuntimeError('Requested device="mps" is not available')
                if self.device == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError('Requested device="cuda" is not available')
                resolved = self.device
            load_kwargs: dict[str, Any] = {'trust_remote_code': True}
            if self.revision is not None:
                load_kwargs['revision'] = self.revision
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            model = model.to(device=resolved, dtype=torch_dtype)
            model.eval()
            self._model = model
            self._resolved_device = resolved
            self._versions = {
                'torch': torch.__version__,
                'transformers': transformers.__version__,
            }
        assert self._resolved_device is not None
        return self._model, torch, self._resolved_device, torch_dtype

    def _seed_torch(self, torch: Any, device: str, seed: int) -> None:
        torch.manual_seed(seed)
        if device == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)
        elif device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def metadata(self) -> dict[str, Any]:
        return {
            'forecaster': self.name,
            'model_id': self.model_id,
            'revision': self.revision,
            'device': self._resolved_device or self.device,
            'dtype': self.dtype,
            'revin': self.revin,
            'sample_chunk_size': self.sample_chunk_size,
            'max_context_length': SUNDIAL_MAX_CONTEXT,
            'pretraining_release_note': (
                'thuml/sundial-base-128m was released February 2025 (arXiv:2502.00816); '
                'check overlap with evaluation dates when interpreting zero-shot results.'
            ),
            'package_versions': dict(self._versions),
        }

    def sample(
        self,
        contexts: np.ndarray,
        horizon: int,
        num_samples: int,
        seed: int,
    ) -> np.ndarray:
        array = np.asarray(contexts, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError('contexts must be a nonempty 2D array (batch_size, context_length)')
        if array.shape[1] > SUNDIAL_MAX_CONTEXT:
            raise ValueError(
                f'Context length {array.shape[1]} exceeds Sundial maximum {SUNDIAL_MAX_CONTEXT}'
            )
        if not np.isfinite(array).all():
            raise ValueError('contexts must contain only finite values')
        model, torch, device, torch_dtype = self._load_model()
        inputs = torch.as_tensor(array, dtype=torch_dtype, device=device)
        chunk = self.sample_chunk_size or num_samples
        chunks = []
        remaining = num_samples
        chunk_index = 0
        with torch.inference_mode():
            while remaining > 0:
                size = min(chunk, remaining)
                self._seed_torch(torch, device, int(seed) + chunk_index)
                output = model.generate(
                    inputs,
                    max_new_tokens=horizon,
                    num_samples=size,
                    revin=self.revin,
                )
                chunks.append(output.detach().to(device='cpu', dtype=torch.float64).numpy())
                remaining -= size
                chunk_index += 1
        return np.concatenate(chunks, axis=1)


class TimestralForecaster:
    """Lazy-loading wrapper around Mistral AI's Timestral quantile-head TSFM."""

    name = 'timestral'

    def __init__(
        self,
        model_id: str = TIMESTRAL_DEFAULT_MODEL_ID,
        device: str = 'cpu',
        dtype: str = 'float32',
        revision: str | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = False,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError('model_id must be a nonempty string')
        _validate_choice('device', device, ('cpu', 'mps', 'cuda', 'auto'))
        _validate_choice('dtype', dtype, ('float32',))
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError('revision must be None or a nonempty string')
        if not isinstance(symmetric_averaging, bool):
            raise ValueError('symmetric_averaging must be a bool')
        if not isinstance(clip_nonnegative, bool):
            raise ValueError('clip_nonnegative must be a bool')
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self._forecaster: Any = None
        self._resolved_device: str | None = None
        self._max_context: int = TIMESTRAL_MAX_CONTEXT
        self._quantile_levels: tuple[float, ...] = TIMESTRAL_QUANTILE_LEVELS
        self._versions: dict[str, str] = {}

    def _load_model(self) -> tuple[Any, Any, str]:
        try:
            import torch
            import timestral
        except ImportError as error:
            raise ImportError(
                'TimestralForecaster requires torch and timestral. '
                'Install the optional extra with `uv sync --extra tsfm` (or '
                'run with `uv run --extra tsfm`).'
            ) from error
        if self._forecaster is None:
            if self.device == 'auto':
                resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                if self.device == 'mps' and not torch.backends.mps.is_available():
                    raise RuntimeError('Requested device="mps" is not available')
                if self.device == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError('Requested device="cuda" is not available')
                resolved = self.device
            load_kwargs: dict[str, Any] = {}
            if self.revision is not None:
                load_kwargs['revision'] = self.revision
            forecaster = timestral.TimestralForecaster.from_pretrained(
                self.model_id,
                device=resolved,
                **load_kwargs,
            )
            self._forecaster = forecaster
            self._resolved_device = resolved
            self._max_context = int(forecaster.max_context)
            self._quantile_levels = tuple(float(level) for level in forecaster.quantile_levels)
            try:
                timestral_version = importlib.metadata.version('timestral')
            except importlib.metadata.PackageNotFoundError:
                timestral_version = 'unknown'
            self._versions = {
                'torch': torch.__version__,
                'timestral': timestral_version,
            }
        assert self._resolved_device is not None
        return self._forecaster, torch, self._resolved_device

    def metadata(self) -> dict[str, Any]:
        return {
            'forecaster': self.name,
            'model_id': self.model_id,
            'revision': self.revision,
            'device': self._resolved_device or self.device,
            'dtype': self.dtype,
            'symmetric_averaging': self.symmetric_averaging,
            'clip_nonnegative': self.clip_nonnegative,
            'max_context_length': self._max_context,
            'quantile_levels': list(self._quantile_levels),
            'pretraining_release_note': (
                'mistralai/Timestral-28M was released October 2026 (v1.0.0); '
                'evaluation windows ending September 2026 predate release.'
            ),
            'package_versions': dict(self._versions),
        }

    def predict_quantiles(
        self,
        contexts: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        horizon_int = _validate_int('horizon', horizon, 1)
        assert horizon_int is not None
        array = np.asarray(contexts, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError('contexts must be a nonempty 2D array (batch_size, context_length)')
        if array.shape[1] > self._max_context:
            raise ValueError(
                f'Context length {array.shape[1]} exceeds Timestral maximum {self._max_context}'
            )
        if not np.isfinite(array).all():
            raise ValueError('contexts must contain only finite values')
        forecaster, torch, _ = self._load_model()
        if array.shape[1] > self._max_context:
            raise ValueError(
                f'Context length {array.shape[1]} exceeds Timestral maximum {self._max_context}'
            )
        output = forecaster.forecast(
            array.astype(np.float32),
            prediction_length=horizon_int,
            symmetric_averaging=self.symmetric_averaging,
            clip_nonnegative=self.clip_nonnegative,
        )
        raw = output.quantiles.detach().to(device='cpu', dtype=torch.float64).numpy()
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f'Unexpected Timestral quantile tensor shape {raw.shape}')
        quantiles = np.ascontiguousarray(raw[:, 0, :, :].transpose(0, 2, 1))
        levels = tuple(float(level) for level in output.quantile_levels)
        return quantiles, levels


@dataclass(frozen=True)
class TSFMConfig:
    """Configuration for 1D portfolio TSFM risk forecasting.

    Weights are never silently normalized. In target='log_price' mode,
    context_sessions complete daily returns produce context_sessions + 1
    log-wealth points anchored at 0 at the forecast origin.
    """

    assets: tuple[str, ...] | None = None
    weights: Mapping[str, float] | None = None
    return_type: str = 'simple'
    target: str = 'log_price'
    horizon_sessions: int = 1
    rebalance: str = 'daily'
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    context_sessions: int = 252
    num_samples: int = 1000
    target_stride: int = 1
    batch_size: int = 1
    seed: int = 0
    forecaster: str = 'sundial'
    model_id: str | None = None
    device: str = 'cpu'
    dtype: str = 'float32'
    sample_chunk_size: int | None = None
    revin: bool | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for name, choices in (
            ('forecaster', tuple(DEFAULT_MODEL_IDS)),
            ('return_type', ('simple', 'log')),
            ('target', ('log_price', 'return')),
            ('rebalance', ('daily',)),
            ('device', ('cpu', 'mps', 'cuda', 'auto')),
            ('dtype', ('float32', 'float16', 'bfloat16')),
        ):
            _validate_choice(name, getattr(self, name), choices)
        for name, minimum, allow_none in (
            ('horizon_sessions', 1, False),
            ('context_sessions', 2, False),
            ('num_samples', 2, False),
            ('target_stride', 1, False),
            ('batch_size', 1, False),
            ('seed', 0, False),
            ('sample_chunk_size', 1, True),
        ):
            object.__setattr__(
                self, name, _validate_int(name, getattr(self, name), minimum, allow_none=allow_none))
        resolved_model_id = (
            DEFAULT_MODEL_IDS[self.forecaster] if self.model_id is None else self.model_id
        )
        if not isinstance(resolved_model_id, str) or not resolved_model_id:
            raise ValueError('model_id must be a nonempty string')
        object.__setattr__(self, 'model_id', resolved_model_id)
        resolved_revin = (self.target != 'log_price') if self.revin is None else self.revin
        if not isinstance(resolved_revin, bool):
            raise ValueError('revin must be a bool or None')
        if self.target == 'log_price' and resolved_revin:
            raise ValueError("revin must be False when target='log_price'")
        object.__setattr__(self, 'revin', resolved_revin)
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError('revision must be None or a nonempty string')
        _validate_assets(self.assets)
        resolved_levels = _validate_confidence_levels(self.confidence_levels)
        object.__setattr__(self, 'confidence_levels', resolved_levels)
        object.__setattr__(self, 'weights', _validate_weights(self.weights, self.assets))
        if self.forecaster == 'timestral':
            if self.dtype != 'float32':
                raise ValueError("TimestralForecaster supports dtype='float32' only")
            if self.target == 'return' and self.horizon_sessions > 1:
                raise ValueError(
                    "Timestral quantile forecasts support target='return' only when horizon_sessions == 1"
                )
            for confidence in resolved_levels:
                target_q = 1.0 - confidence
                if not any(
                    abs(target_q - grid_q) <= QUANTILE_LEVEL_TOLERANCE
                    for grid_q in TIMESTRAL_QUANTILE_LEVELS
                ):
                    raise ValueError(
                        f'confidence level {confidence} requires quantile {target_q:g}, which is not on '
                        f'Timestral grid {TIMESTRAL_QUANTILE_LEVELS}'
                    )


@dataclass(frozen=True)
class TSFMSummary:
    """Single-origin TSFM risk summary, simulated scenarios or quantiles, and provenance."""

    risk: pd.DataFrame
    scenarios: pd.DataFrame
    metadata: dict[str, Any]
    quantiles: pd.DataFrame | None = None


@dataclass(frozen=True)
class TSFMBacktest:
    """Chronological TSFM forecasts and descriptive coverage; no independence tests."""

    forecasts: pd.DataFrame
    coverage: pd.DataFrame
    metadata: dict[str, Any]


class TSFMForecastBenchmark:
    """Forecast a 1D portfolio series with a probabilistic TSFM and estimate VaR/ES.

    No method writes files. Fitting validates data only and never loads PyTorch.
    Backtest calls are strictly causal and never mutate fitted state.
    """

    def __init__(
        self,
        config: TSFMConfig | None = None,
        forecaster: Forecaster | None = None,
    ) -> None:
        self.config = TSFMConfig() if config is None else config
        if forecaster is not None:
            resolved_forecaster: Forecaster = forecaster
        elif self.config.forecaster == 'timestral':
            assert self.config.model_id is not None
            resolved_forecaster = TimestralForecaster(
                model_id=self.config.model_id,
                device=self.config.device,
                dtype=self.config.dtype,
                revision=self.config.revision,
            )
        else:
            assert self.config.model_id is not None
            assert self.config.revin is not None
            resolved_forecaster = SundialForecaster(
                model_id=self.config.model_id,
                device=self.config.device,
                dtype=self.config.dtype,
                sample_chunk_size=self.config.sample_chunk_size,
                revin=self.config.revin,
                revision=self.config.revision,
            )
        has_sample = callable(getattr(resolved_forecaster, 'sample', None))
        has_quantiles = callable(getattr(resolved_forecaster, 'predict_quantiles', None))
        if has_sample == has_quantiles:
            raise TypeError(
                'forecaster must implement either SampleForecaster.sample() '
                'or QuantileForecaster.predict_quantiles()'
            )
        self._forecaster: Forecaster = resolved_forecaster
        self._quantile_mode: bool = has_quantiles
        if self._quantile_mode and self.config.target == 'return' and self.config.horizon_sessions > 1:
            raise ValueError(
                "Quantile forecasters support target='return' only when horizon_sessions == 1"
            )
        self._history: pd.DataFrame | None = None
        self._assets: tuple[str, ...] = ()
        self._weights = np.empty(0, dtype=np.float64)
        self._fit_info: dict[str, Any] = {}

    def fit(self, returns: ReturnInput) -> 'TSFMForecastBenchmark':
        """Store validated historical returns and portfolio weights; return self."""
        history, info = _load(returns, self.config.assets, self.config.return_type)
        if history.empty:
            raise ValueError('Fitted history must contain at least one session')
        assets = tuple(history.columns)
        weights = _resolve_weights(assets, self.config.weights)
        self._history, self._fit_info = history, info
        self._assets, self._weights = assets, weights
        return self

    def _require_history(self) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError('Fit TSFMForecastBenchmark before summarizing or backtesting')
        return self._history

    def _build_context(self, history: pd.DataFrame) -> tuple[np.ndarray | None, dict[str, Any], str]:
        """Build one causal context vector from the last context_sessions complete rows."""
        values = history.to_numpy(dtype=np.float64)
        complete_idx = np.flatnonzero(~np.isnan(values).any(axis=1))
        required = self.config.context_sessions
        if len(complete_idx) < required:
            info = {
                'history_start': None,
                'history_end': None,
                'n_history_sessions': len(history),
                'n_context_sessions': int(len(complete_idx)),
                'n_excluded_context_sessions': int(len(history) - len(complete_idx)),
            }
            return None, info, 'insufficient_history'
        selected = complete_idx[-required:]
        first_pos, last_pos = int(selected[0]), int(selected[-1])
        span_sessions = len(history) - first_pos
        info = {
            'history_start': history.index[first_pos].date().isoformat(),
            'history_end': history.index[last_pos].date().isoformat(),
            'n_history_sessions': int(span_sessions),
            'n_context_sessions': int(required),
            'n_excluded_context_sessions': int(span_sessions - required),
        }
        with np.errstate(over='raise', invalid='raise'):
            port_returns = values[selected] @ self._weights
        if not np.isfinite(port_returns).all():
            raise FloatingPointError('Portfolio context return calculation produced nonfinite values')
        if self.config.target == 'log_price':
            if np.any(port_returns <= -1.0):
                return None, info, 'invalid_log_price_context'
            log_rets = np.log1p(port_returns)
            context = np.empty(required + 1, dtype=np.float64)
            context[:-1] = -np.cumsum(log_rets[::-1])[::-1]
            context[-1] = 0.0
            return context, info, 'ok'
        return port_returns.astype(np.float64, copy=True), info, 'ok'

    def _paths_to_returns(self, paths: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert sampled trajectories (batch, num_samples, horizon) to H-session returns."""
        array = np.asarray(paths, dtype=np.float64)
        expected = (batch_size, self.config.num_samples, self.config.horizon_sessions)
        if array.shape != expected:
            raise ValueError(f'Forecaster returned shape {array.shape}, expected {expected}')
        if not np.isfinite(array).all():
            raise FloatingPointError('Forecaster produced nonfinite sample paths')
        if self.config.target == 'log_price':
            terminal_log = array[:, :, -1]
            with np.errstate(over='raise', invalid='raise'):
                returns = np.expm1(terminal_log)
            if not np.isfinite(returns).all():
                raise FloatingPointError('Log-price return conversion produced nonfinite values')
            return returns, np.zeros(batch_size, dtype=int)
        factors = 1.0 + array
        wiped_out = np.any(factors <= 0.0, axis=2)
        safe_factors = np.where(wiped_out[:, :, None], 1.0, factors)
        with np.errstate(over='raise', invalid='raise'):
            returns = np.prod(safe_factors, axis=2) - 1.0
        returns = np.where(wiped_out, -1.0, returns)
        if not np.isfinite(returns).all():
            raise FloatingPointError('Simulated return compounding produced nonfinite values')
        clamped = wiped_out.sum(axis=1).astype(int)
        return returns, clamped

    def _resolve_quantile_indices(self, levels: Sequence[float]) -> tuple[tuple[float, ...], list[int]]:
        """Validate returned quantile levels and locate each exact 1 - confidence level."""
        level_tuple = tuple(float(level) for level in levels)
        if (not level_tuple
                or any(not np.isfinite(q) or not 0.0 < q < 1.0 for q in level_tuple)
                or any(a >= b for a, b in zip(level_tuple, level_tuple[1:]))):
            raise ValueError('Forecaster quantile levels must be strictly increasing finite values in (0, 1)')
        indices: list[int] = []
        for confidence in self.config.confidence_levels:
            target_q = 1.0 - confidence
            matches = [i for i, q in enumerate(level_tuple) if abs(q - target_q) <= QUANTILE_LEVEL_TOLERANCE]
            if not matches:
                raise ValueError(
                    f'Forecaster quantile levels {level_tuple} do not include required level '
                    f'{target_q:g} for confidence {confidence}'
                )
            indices.append(matches[0])
        return level_tuple, indices

    def _quantiles_to_vars(
        self,
        quantiles: np.ndarray,
        levels: Sequence[float],
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
        """Convert quantile trajectories (batch, n_levels, horizon) to terminal VaRs."""
        level_tuple, conf_indices = self._resolve_quantile_indices(levels)
        array = np.asarray(quantiles, dtype=np.float64)
        expected = (batch_size, len(level_tuple), self.config.horizon_sessions)
        if array.shape != expected:
            raise ValueError(f'Forecaster returned quantile shape {array.shape}, expected {expected}')
        if not np.isfinite(array).all():
            raise FloatingPointError('Forecaster produced nonfinite quantile forecasts')
        if len(level_tuple) > 1 and np.any(np.diff(array, axis=1) < -QUANTILE_CROSSING_TOLERANCE):
            raise ValueError('Forecaster produced crossing quantiles')
        terminal = array[:, :, -1]
        if self.config.target == 'log_price':
            with np.errstate(over='raise', invalid='raise'):
                terminal_returns = np.expm1(terminal)
            if not np.isfinite(terminal_returns).all():
                raise FloatingPointError('Log-price quantile conversion produced nonfinite values')
            clamped = np.zeros(batch_size, dtype=int)
        else:
            if self.config.horizon_sessions != 1:
                raise ValueError(
                    "Quantile forecasters support target='return' only when horizon_sessions == 1"
                )
            wiped_out = terminal <= -1.0
            terminal_returns = np.where(wiped_out, -1.0, terminal)
            clamped = wiped_out[:, conf_indices].sum(axis=1).astype(int)
        var_matrix = -terminal_returns[:, conf_indices]
        return var_matrix, clamped, terminal_returns, level_tuple

    def _risk_rows(self, losses: np.ndarray | None, n_clamped: int) -> list[dict[str, Any]]:
        available = losses is not None
        n = len(losses) if available else 0
        sorted_losses = np.sort(losses) if available else np.empty(0, dtype=np.float64)
        records = []
        for confidence in self.config.confidence_levels:
            mass = _tail_mass(n, confidence) if available else 0.0
            records.append({
                'method': 'tsfm',
                'forecaster': self._forecaster.name,
                'target': self.config.target,
                'confidence': confidence,
                'horizon_sessions': self.config.horizon_sessions,
                'rebalance': self.config.rebalance,
                'var': _empirical_var(sorted_losses, confidence) if available else np.nan,
                'es': _empirical_es(sorted_losses, confidence) if available else np.nan,
                'tail_mass_observations': mass,
                'small_tail_sample': mass < 20,
                'overlapping_scenarios': False,
                'n_scenarios': n,
                'n_clamped_paths': int(n_clamped),
            })
        return records

    def _quantile_risk_rows(self, var_values: np.ndarray | None, n_clamped: int) -> list[dict[str, Any]]:
        available = var_values is not None
        records = []
        for i, confidence in enumerate(self.config.confidence_levels):
            records.append({
                'method': 'tsfm',
                'forecaster': self._forecaster.name,
                'target': self.config.target,
                'confidence': confidence,
                'horizon_sessions': self.config.horizon_sessions,
                'rebalance': self.config.rebalance,
                'var': float(var_values[i]) if available else np.nan,
                'es': np.nan,
                'tail_mass_observations': np.nan,
                'small_tail_sample': pd.NA,
                'overlapping_scenarios': False,
                'n_scenarios': 0,
                'n_clamped_paths': int(n_clamped),
            })
        return records

    def _metadata(
        self,
        counts: dict[str, Any],
        evaluation: dict[str, Any] | None = None,
        quantile_levels: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        config = {name: getattr(self.config, name) for name in self.config.__dataclass_fields__
                  if name != 'weights'}
        config['assets'] = list(self._assets)
        config['confidence_levels'] = list(self.config.confidence_levels)
        config['weights'] = dict(zip(self._assets, self._weights.tolist()))
        forecaster_meta = self._forecaster.metadata()
        packages = {'python': platform.python_version(),
                    'numpy': np.__version__, 'pandas': pd.__version__,
                    **forecaster_meta.get('package_versions', {})}
        meta: dict[str, Any] = {
            'method': 'tsfm',
            'forecast_type': 'quantiles' if self._quantile_mode else 'samples',
            'forecaster': forecaster_meta,
            'configuration': config,
            'input_return_type': self.config.return_type,
            'target_representation': self.config.target,
            'context_vector_length': (
                self.config.context_sessions + 1
                if self.config.target == 'log_price' else self.config.context_sessions
            ),
            'loss_units': 'fractional',
            'inputs': {'fit': dict(self._fit_info), **({'evaluation': evaluation}
                       if evaluation is not None else {})},
            'counts': counts,
            'package_versions': packages,
            'missing_context_policy': (
                'select the last context_sessions complete portfolio sessions strictly before '
                'target_start; skip incomplete rows without filling; require all selected assets'
            ),
            'return_clamping_policy': (
                'limited liability floor at return=-1.0 if any simulated daily return <= -1.0 '
                '(target="return" only); logged in n_clamped_paths'
            ),
            'quantile_convention': (
                'exact forecast quantile grid level (1 - confidence); no interpolation'
                if self._quantile_mode else 'inverse empirical CDF; no interpolation'
            ),
            'es_convention': (
                'not available (NaN); quantile head does not resolve tail below lowest grid level'
                if self._quantile_mode else 'mean of worst 1-confidence mass with fractional boundary'
            ),
            'es_integer_mass_tolerance_per_scenario': ES_INTEGER_TOLERANCE,
            'weight_sum_absolute_tolerance': WEIGHT_TOLERANCE,
            'target_stride': self.config.target_stride,
            'overlapping_scenarios': False,
            'overlapping_targets': self.config.horizon_sessions > self.config.target_stride,
        }
        if self._quantile_mode:
            levels_out = (
                list(quantile_levels)
                if quantile_levels is not None else forecaster_meta.get('quantile_levels')
            )
            meta['quantile_levels'] = levels_out
        return meta

    def summarize(self) -> TSFMSummary:
        """Forecast from the end of fitted history; fail explicitly if context is unavailable."""
        history = self._require_history()
        context, info, status = self._build_context(history)
        if context is None:
            raise ValueError(
                f'Unavailable context ({status}): assets={self._assets}, '
                f'context_sessions={self.config.context_sessions}, {info}'
            )
        if self._quantile_mode:
            raw_q, levels = self._forecaster.predict_quantiles(  # type: ignore[union-attr]
                context[None, :],
                horizon=self.config.horizon_sessions,
            )
            var_matrix, clamped, terminal_returns, level_tuple = self._quantiles_to_vars(
                raw_q, levels, batch_size=1,
            )
            scenarios = pd.DataFrame({
                'sample_index': pd.Series(dtype=int),
                'portfolio_return': pd.Series(dtype=float),
                'loss': pd.Series(dtype=float),
            })
            quantiles_df = pd.DataFrame({
                'quantile_level': np.asarray(level_tuple, dtype=np.float64),
                'portfolio_return': terminal_returns[0],
                'loss': -terminal_returns[0],
            })
            records = [{**risk, **info,
                        'forecast_origin': history.index[-1],
                        'portfolio_mean_return': np.nan, 'portfolio_std_return': np.nan}
                       for risk in self._quantile_risk_rows(var_matrix[0], int(clamped[0]))]
            risk = _dated_table(records, list(records[0]),
                                ('history_start', 'history_end', 'forecast_origin'))
            risk['small_tail_sample'] = risk['small_tail_sample'].astype('boolean')
            counts = {**info, 'n_scenarios': 0, 'n_quantile_levels': len(level_tuple),
                      'n_clamped_paths': int(clamped[0])}
            return TSFMSummary(
                risk, scenarios, self._metadata(counts, quantile_levels=level_tuple),
                quantiles=quantiles_df,
            )

        paths = self._forecaster.sample(  # type: ignore[union-attr]
            context[None, :],
            horizon=self.config.horizon_sessions,
            num_samples=self.config.num_samples,
            seed=self.config.seed,
        )
        returns, clamped = self._paths_to_returns(paths, batch_size=1)
        sample_returns = returns[0]
        losses = -sample_returns
        with np.errstate(over='raise', invalid='raise'):
            mean = float(sample_returns.mean())
            std = float(sample_returns.std(ddof=1))
        if not np.isfinite([mean, std]).all():
            raise FloatingPointError('Portfolio scenario statistics are nonfinite')
        scenarios = pd.DataFrame({
            'sample_index': np.arange(self.config.num_samples, dtype=int),
            'portfolio_return': sample_returns,
            'loss': losses,
        })
        records = [{**risk, **info,
                    'forecast_origin': history.index[-1],
                    'portfolio_mean_return': mean, 'portfolio_std_return': std}
                   for risk in self._risk_rows(losses, int(clamped[0]))]
        risk = _dated_table(records, list(records[0]),
                            ('history_start', 'history_end', 'forecast_origin'))
        counts = {**info, 'n_scenarios': self.config.num_samples,
                  'n_clamped_paths': int(clamped[0])}
        return TSFMSummary(risk, scenarios, self._metadata(counts))

    def backtest(self, evaluation_returns: ReturnInput) -> TSFMBacktest:
        """Walk-forward forecast evaluation targets strictly without future inputs."""
        history = self._require_history()
        evaluation, info = _load(evaluation_returns, self._assets, self.config.return_type)
        if len(evaluation) and evaluation.index[0] <= history.index[-1]:
            raise ValueError('Evaluation dates must be strictly after fitted history')
        combined = pd.concat([history, evaluation])
        horizon = self.config.horizon_sessions
        n_available_starts = max(0, len(evaluation) - horizon + 1)
        starts = list(range(0, n_available_starts, self.config.target_stride))
        n_targets = len(starts)

        contexts: list[np.ndarray] = []
        valid_positions: list[int] = []
        context_infos: list[dict[str, Any]] = []
        statuses: list[str] = []
        for idx, start in enumerate(starts):
            position = len(history) + start
            context, ctx_info, status = self._build_context(combined.iloc[:position])
            context_infos.append(ctx_info)
            statuses.append(status)
            if context is not None:
                contexts.append(context)
                valid_positions.append(idx)

        sampled_losses: dict[int, np.ndarray] = {}
        quantile_vars: dict[int, np.ndarray] = {}
        observed_levels: tuple[float, ...] | None = None
        clamped_counts: dict[int, int] = {}
        step = self.config.batch_size
        for offset in range(0, len(contexts), step):
            batch_contexts = np.stack(contexts[offset:offset + step], axis=0)
            batch_indices = valid_positions[offset:offset + step]
            if self._quantile_mode:
                raw_q, levels = self._forecaster.predict_quantiles(  # type: ignore[union-attr]
                    batch_contexts,
                    horizon=horizon,
                )
                var_matrix, clamped, _, level_tuple = self._quantiles_to_vars(
                    raw_q, levels, batch_size=len(batch_indices),
                )
                observed_levels = level_tuple
                for local_i, target_i in enumerate(batch_indices):
                    quantile_vars[target_i] = var_matrix[local_i]
                    clamped_counts[target_i] = int(clamped[local_i])
            else:
                paths = self._forecaster.sample(  # type: ignore[union-attr]
                    batch_contexts,
                    horizon=horizon,
                    num_samples=self.config.num_samples,
                    seed=self.config.seed + batch_indices[0],
                )
                returns, clamped = self._paths_to_returns(paths, batch_size=len(batch_indices))
                for local_i, target_i in enumerate(batch_indices):
                    sampled_losses[target_i] = -returns[local_i]
                    clamped_counts[target_i] = int(clamped[local_i])

        records = []
        for idx, start in enumerate(starts):
            position = len(history) + start
            block = evaluation.iloc[start:start + horizon].to_numpy()
            realized_ok = not np.isnan(block).any()
            realized = (float(_portfolio_returns(block[None], self._weights,
                                                 self.config.rebalance)[0])
                        if realized_ok else np.nan)
            estimate_ok = statuses[idx] == 'ok'
            if self._quantile_mode:
                risks = self._quantile_risk_rows(
                    quantile_vars[idx] if estimate_ok else None,
                    clamped_counts.get(idx, 0),
                )
            else:
                risks = self._risk_rows(
                    sampled_losses[idx] if estimate_ok else None,
                    clamped_counts.get(idx, 0),
                )
            for risk in risks:
                records.append({
                    **risk,
                    **context_infos[idx],
                    'forecast_origin': combined.index[position - 1],
                    'target_start': evaluation.index[start],
                    'target_end': evaluation.index[start + horizon - 1],
                    'realized_return': realized,
                    'realized_loss': -realized,
                    'estimate_status': statuses[idx],
                    'realization_status': 'ok' if realized_ok else 'missing_returns',
                    'breach': bool(-realized > risk['var']) if estimate_ok and realized_ok else pd.NA,
                })

        columns = [
            'method', 'forecaster', 'target', 'confidence', 'horizon_sessions', 'rebalance',
            'var', 'es', 'tail_mass_observations', 'small_tail_sample', 'overlapping_scenarios',
            'n_scenarios', 'n_clamped_paths', 'history_start', 'history_end',
            'n_history_sessions', 'n_context_sessions', 'n_excluded_context_sessions',
            'forecast_origin', 'target_start', 'target_end', 'realized_return',
            'realized_loss', 'estimate_status', 'realization_status', 'breach',
        ]
        forecasts = _dated_table(records, columns, (
            'history_start', 'history_end', 'forecast_origin', 'target_start', 'target_end'))
        forecasts['breach'] = forecasts.breach.astype('boolean')
        if self._quantile_mode:
            forecasts['small_tail_sample'] = forecasts['small_tail_sample'].astype('boolean')
        coverage = _coverage_table(forecasts, self.config.confidence_levels, horizon, n_targets)
        first_conf = forecasts.loc[forecasts.confidence.eq(self.config.confidence_levels[0])]
        counts = {
            'n_available_starts': n_available_starts,
            'n_targets': n_targets,
            'n_trailing_excluded': len(evaluation) - n_available_starts,
            'n_forecast_rows': len(forecasts),
            'n_unavailable_estimate_targets': int(first_conf.estimate_status.ne('ok').sum()),
            'n_missing_realization_targets': int(
                first_conf.realization_status.eq('missing_returns').sum()),
            'total_clamped_paths': int(first_conf.n_clamped_paths.sum()) if len(first_conf) else 0,
        }
        return TSFMBacktest(
            forecasts, coverage,
            self._metadata(counts, info, quantile_levels=observed_levels),
        )
