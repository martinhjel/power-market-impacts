from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass
class MarketStepRecord:
    name: str
    busbar: str
    price: float
    capacity: pd.DataFrame
    fuel_type: str = ""


def object_name(obj) -> str:
    return str(getattr(obj, "name", obj))


def object_busbar_name(obj) -> str | None:
    busbar_name = getattr(obj, "busbar_name", None)
    if busbar_name:
        return str(busbar_name)

    name = object_name(obj)
    parts = name.split("_")
    if parts and parts[0] == "Wind" and len(parts) > 1:
        return parts[1]
    if parts:
        return parts[0]
    return None


def align_to_base(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.shape[1] == 1 and len(base.columns) > 1:
        out = pd.concat([out.iloc[:, 0]] * len(base.columns), axis=1)
        out.columns = base.columns
    if isinstance(out.index, pd.DatetimeIndex) and isinstance(base.index, pd.DatetimeIndex):
        target_tz = base.index.tz
        if target_tz is not None:
            if out.index.tz is None:
                out.index = out.index.tz_localize(target_tz)
            elif out.index.tz != target_tz:
                out.index = out.index.tz_convert(target_tz)
        elif out.index.tz is not None:
            out.index = out.index.tz_convert(None)
    return out.reindex(index=base.index, columns=base.columns, fill_value=0.0).astype("float32")


def capacity_to_base_axes(obj, base: pd.DataFrame) -> pd.DataFrame:
    try:
        data = np.asarray(obj.capacity.scenarios, dtype=float)
    except Exception:
        return pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")

    if data.size == 0:
        return pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")

    if data.size == 1:
        return pd.DataFrame(float(data.ravel()[0]), index=base.index, columns=base.columns, dtype="float32")

    timestamps = getattr(obj.capacity, "timestamps", None)
    if timestamps is not None:
        timestamps = np.asarray(timestamps).ravel()
        if data.ndim == 1 and timestamps.size == data.size:
            series = pd.Series(data.ravel(), index=pd.to_datetime(timestamps))
            df = pd.concat([series] * len(base.columns), axis=1)
            df.columns = base.columns
            return align_to_base(df, base)
        if data.ndim > 1 and data.shape[-1] == timestamps.size:
            df = pd.DataFrame(data.reshape(-1, timestamps.size), columns=pd.to_datetime(timestamps)).T
            df.columns = list(base.columns[: df.shape[1]])
            return align_to_base(df, base)

    if data.ndim == 1:
        if data.size < len(base.index):
            repeats = int(np.ceil(len(base.index) / data.size))
            data = np.tile(data, repeats)
        tiled = np.tile(data[: len(base.index), None], (1, len(base.columns)))
        return pd.DataFrame(tiled, index=base.index, columns=base.columns, dtype="float32")

    if data.shape == base.shape:
        return pd.DataFrame(data, index=base.index, columns=base.columns, dtype="float32")

    if data.shape[0] == len(base.index):
        columns = list(base.columns[: data.shape[1]])
        df = pd.DataFrame(data, index=base.index, columns=columns)
        return align_to_base(df, base)

    if data.shape[1] == len(base.index):
        data = data.T
        columns = list(base.columns[: data.shape[1]])
        df = pd.DataFrame(data, index=base.index, columns=columns)
        return align_to_base(df, base)

    return pd.DataFrame(float(data.ravel()[0]), index=base.index, columns=base.columns, dtype="float32")


def market_step_price(obj) -> float:
    try:
        return float(np.asarray(obj.price.scenarios).ravel()[0])
    except Exception:
        return 0.0


def market_step_fuel_type(obj) -> str:
    return str(getattr(obj, "fuel_type", "") or "")


def collect_market_step_records(model, area: str, base: pd.DataFrame) -> list[MarketStepRecord]:
    records = []
    for market_step in model.market_steps():
        if getattr(market_step, "busbar_name", None) != area:
            continue
        records.append(
            MarketStepRecord(
                name=object_name(market_step),
                busbar=area,
                price=market_step_price(market_step),
                capacity=capacity_to_base_axes(market_step, base),
                fuel_type=market_step_fuel_type(market_step),
            )
        )
    return records


def reconstruct_named_market_step_dispatch(
    *,
    market_steps: Iterable[MarketStepRecord],
    market_price: pd.DataFrame,
    total_market_steps: pd.DataFrame,
    is_target: Callable[[MarketStepRecord], bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = market_price
    target_dispatch = pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")
    target_available = pd.DataFrame(0.0, index=base.index, columns=base.columns, dtype="float32")
    remaining = total_market_steps.reindex(index=base.index, columns=base.columns, fill_value=0.0).astype("float32")

    for record in sorted(market_steps, key=lambda item: (item.price, item.name)):
        capacity = record.capacity.reindex(index=base.index, columns=base.columns, fill_value=0.0).astype("float32")
        if is_target(record):
            target_available = target_available.add(capacity, fill_value=0.0)

        eligible_capacity = capacity.where(market_price >= record.price, 0.0)
        take = remaining.clip(lower=0.0)
        take = take.where(take <= eligible_capacity, eligible_capacity)

        if is_target(record):
            target_dispatch = target_dispatch.add(take, fill_value=0.0)

        remaining = remaining - take

    return target_dispatch.astype("float32"), target_available.astype("float32")
