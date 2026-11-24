"""Zero-shot Time Series Foundation Model (TSFM) portfolio risk benchmark.

Forecasts a 1D daily-rebalanced portfolio series (cumulative log-wealth or
daily returns) strictly causally and computes empirical VaR and ES from
generated sample trajectories using the same conventions as historical
simulation. Deep-learning dependencies are imported lazily only when a
concrete neural forecaster is invoked.
"""

from collections.abc import Mapping
from dataclasses import dataclass
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
    model_id: str = SUNDIAL_DEFAULT_MODEL_ID
    device: str = 'cpu'
    dtype: str = 'float32'
    sample_chunk_size: int | None = None
    revin: bool = True
    revision: str | None = None

    def __post_init__(self) -> None:
        for name, choices in (
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
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError('model_id must be a nonempty string')
        if not isinstance(self.revin, bool):
            raise ValueError('revin must be a bool')
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError('revision must be None or a nonempty string')
        _validate_assets(self.assets)
        object.__setattr__(self, 'confidence_levels',
                           _validate_confidence_levels(self.confidence_levels))
        object.__setattr__(self, 'weights', _validate_weights(self.weights, self.assets))


@dataclass(frozen=True)
class TSFMSummary:
    """Single-origin TSFM risk summary, simulated portfolio scenarios, and provenance."""

    risk: pd.DataFrame
    scenarios: pd.DataFrame
    metadata: dict[str, Any]


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
        forecaster: SampleForecaster | None = None,
    ) -> None:
        self.config = TSFMConfig() if config is None else config
        self._forecaster: SampleForecaster = (
            forecaster if forecaster is not None else SundialForecaster(
                model_id=self.config.model_id,
                device=self.config.device,
                dtype=self.config.dtype,
                sample_chunk_size=self.config.sample_chunk_size,
                revin=self.config.revin,
                revision=self.config.revision,
            )
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

    def _metadata(self, counts: dict[str, Any],
                  evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
        config = {name: getattr(self.config, name) for name in self.config.__dataclass_fields__
                  if name != 'weights'}
        config['assets'] = list(self._assets)
        config['confidence_levels'] = list(self.config.confidence_levels)
        config['weights'] = dict(zip(self._assets, self._weights.tolist()))
        forecaster_meta = self._forecaster.metadata()
        packages = {'python': platform.python_version(),
                    'numpy': np.__version__, 'pandas': pd.__version__,
                    **forecaster_meta.get('package_versions', {})}
        return {
            'method': 'tsfm',
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
            'quantile_convention': 'inverse empirical CDF; no interpolation',
            'es_convention': 'mean of worst 1-confidence mass with fractional boundary',
            'es_integer_mass_tolerance_per_scenario': ES_INTEGER_TOLERANCE,
            'weight_sum_absolute_tolerance': WEIGHT_TOLERANCE,
            'target_stride': self.config.target_stride,
            'overlapping_scenarios': False,
            'overlapping_targets': self.config.horizon_sessions > self.config.target_stride,
        }

    def summarize(self) -> TSFMSummary:
        """Forecast from the end of fitted history; fail explicitly if context is unavailable."""
        history = self._require_history()
        context, info, status = self._build_context(history)
        if context is None:
            raise ValueError(
                f'Unavailable context ({status}): assets={self._assets}, '
                f'context_sessions={self.config.context_sessions}, {info}'
            )
        paths = self._forecaster.sample(
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
        clamped_counts: dict[int, int] = {}
        step = self.config.batch_size
        for offset in range(0, len(contexts), step):
            batch_contexts = np.stack(contexts[offset:offset + step], axis=0)
            batch_indices = valid_positions[offset:offset + step]
            paths = self._forecaster.sample(
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
        return TSFMBacktest(forecasts, coverage, self._metadata(counts, info))
