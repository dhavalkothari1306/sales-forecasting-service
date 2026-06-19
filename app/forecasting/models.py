from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import joblib
import numpy as np
import pandas as pd

from app.forecasting.features import (
    build_future_feature_row,
    create_supervised_frame,
    feature_columns,
    make_lstm_sequences,
)


@dataclass
class ForecastResult:
    model_name: str
    predictions: np.ndarray


class ForecastModel(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> "ForecastModel":
        ...

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, output_dir: Path) -> None:
        ...


class BaseModel:
    name = "base"

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "model.joblib")


class SeasonalNaiveModel(BaseModel):
    """Safety baseline used only when every mandatory candidate fails."""

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 52, freq: str = "W-SUN") -> None:
        self.seasonal_period = seasonal_period
        self.freq = freq
        self.history: list[float] = []

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaiveModel":
        self.history = train_df.sort_values("date")["y"].astype(float).tolist()
        return self

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        history = history_df.sort_values("date")["y"].astype(float).tolist()
        preds: list[float] = []
        for step in range(horizon):
            if len(history) >= self.seasonal_period:
                pred = history[-self.seasonal_period]
            else:
                pred = float(np.mean(history[-min(len(history), 8) :]))
            pred = max(0.0, float(pred))
            preds.append(pred)
            history.append(pred)
        return np.array(preds, dtype=float)


class SARIMAModel(BaseModel):
    name = "sarima"

    def __init__(self, seasonal_period: int = 52, freq: str = "W-SUN") -> None:
        self.seasonal_period = seasonal_period
        self.freq = freq
        self.result = None

    def fit(self, train_df: pd.DataFrame) -> "SARIMAModel":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError("statsmodels is required for SARIMA. Install requirements.txt.") from exc

        y = train_df.sort_values("date")["y"].astype(float).to_numpy()
        if len(y) < 40:
            raise ValueError("SARIMA needs at least 40 weekly observations.")

        seasonal_order = (
            (1, 0, 1, self.seasonal_period)
            if len(y) >= self.seasonal_period * 2
            else (0, 0, 0, 0)
        )
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit(disp=False, maxiter=100)
        return self

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("SARIMA model has not been fitted.")
        return np.clip(np.asarray(self.result.forecast(steps=horizon), dtype=float), 0.0, None)


class ProphetModel(BaseModel):
    name = "prophet"

    def __init__(self, freq: str = "W-SUN") -> None:
        self.freq = freq
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> "ProphetModel":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("prophet is required for Prophet. Install requirements.txt.") from exc

        prophet_df = (
            train_df.sort_values("date")[["date", "y"]]
            .rename(columns={"date": "ds", "y": "y"})
            .reset_index(drop=True)
        )
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.fit(prophet_df)
        self.model = model
        return self

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Prophet model has not been fitted.")
        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq, include_history=False)
        forecast = self.model.predict(future)
        return np.clip(forecast["yhat"].to_numpy(dtype=float), 0.0, None)


class XGBoostLagModel(BaseModel):
    name = "xgboost"

    def __init__(self, freq: str = "W-SUN", random_seed: int = 42) -> None:
        self.freq = freq
        self.random_seed = random_seed
        self.model = None
        self.feature_columns = feature_columns()

    def fit(self, train_df: pd.DataFrame) -> "XGBoostLagModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for the lag feature model. Install requirements.txt.") from exc

        supervised = create_supervised_frame(train_df, drop_missing=True)
        if len(supervised) < 20:
            raise ValueError("XGBoost needs at least 20 supervised rows after lag creation.")

        x_train = supervised[self.feature_columns]
        y_train = supervised["y"]
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_seed,
            n_jobs=1,
        )
        self.model.fit(x_train, y_train)
        return self

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoost model has not been fitted.")

        history = history_df.sort_values("date")["y"].astype(float).tolist()
        last_date = pd.Timestamp(history_df["date"].max())
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq=self.freq)[1:]
        predictions: list[float] = []
        for future_date in future_dates:
            row = build_future_feature_row(history, future_date)
            pred = float(self.model.predict(row[self.feature_columns])[0])
            pred = max(0.0, pred)
            predictions.append(pred)
            history.append(pred)
        return np.array(predictions, dtype=float)


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(
        self,
        sequence_length: int = 30,
        epochs: int = 25,
        batch_size: int = 16,
        random_seed: int = 42,
        freq: str = "W-SUN",
    ) -> None:
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.freq = freq
        self.model = None
        self.scaler = None

    def fit(self, train_df: pd.DataFrame) -> "LSTMModel":
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras import callbacks, layers, models
        except ImportError as exc:
            raise ImportError("tensorflow and scikit-learn are required for LSTM.") from exc

        tf.keras.utils.set_random_seed(self.random_seed)
        values = train_df.sort_values("date")["y"].astype(float).to_numpy().reshape(-1, 1)
        if len(values) <= self.sequence_length + 5:
            raise ValueError("LSTM needs more observations than the configured sequence length.")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
        x_train, y_train = make_lstm_sequences(scaled, self.sequence_length)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        model = models.Sequential(
            [
                layers.Input(shape=(self.sequence_length, 1)),
                layers.LSTM(32),
                layers.Dropout(0.1),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
        validation_split = 0.15 if len(x_train) >= 50 else 0.0
        callback_list = [early_stop] if validation_split else []
        model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=0,
            shuffle=False,
        )
        self.model = model
        self.scaler = scaler
        return self

    def predict(self, horizon: int, history_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("LSTM model has not been fitted.")

        values = history_df.sort_values("date")["y"].astype(float).to_numpy().reshape(-1, 1)
        scaled_history = self.scaler.transform(values).reshape(-1).tolist()
        predictions: list[float] = []
        for _ in range(horizon):
            window = np.array(scaled_history[-self.sequence_length :], dtype=float).reshape(
                1, self.sequence_length, 1
            )
            scaled_pred = float(self.model.predict(window, verbose=0)[0][0])
            scaled_history.append(scaled_pred)
            pred = float(self.scaler.inverse_transform([[scaled_pred]])[0][0])
            predictions.append(max(0.0, pred))
        return np.array(predictions, dtype=float)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.model is None or self.scaler is None:
            raise RuntimeError("Cannot save an unfitted LSTM model.")
        model_dir = output_dir / "keras_model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        self.model.save(output_dir / "lstm.keras")
        joblib.dump(
            {
                "scaler": self.scaler,
                "sequence_length": self.sequence_length,
                "freq": self.freq,
            },
            output_dir / "preprocessor.joblib",
        )


def model_factories(freq: str, lstm_epochs: int, random_seed: int) -> dict[str, Callable[[], ForecastModel]]:
    return {
        "sarima": lambda: SARIMAModel(freq=freq),
        "prophet": lambda: ProphetModel(freq=freq),
        "xgboost": lambda: XGBoostLagModel(freq=freq, random_seed=random_seed),
        "lstm": lambda: LSTMModel(freq=freq, epochs=lstm_epochs, random_seed=random_seed),
    }
