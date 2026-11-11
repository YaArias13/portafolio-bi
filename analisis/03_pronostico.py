"""
Pronóstico de ventas mensuales.

Pregunta: ¿cuánto se va a vender en los próximos seis meses, y qué tan seguro es ese número?

Pasos:
    1. Prueba hacia atrás: se entrena con los primeros 36 meses y se compara
       contra los 12 siguientes, que el modelo no vio.
    2. Se compara Holt-Winters contra la regla simple "lo mismo que el año pasado".
    3. Se reentrena con los 48 meses y se proyectan 6, con un rango del 80 %.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATOS = Path(__file__).resolve().parent / "datos_simulados" / "ventas_mensuales.csv"
HORIZONTE, PRUEBA = 6, 12
Z_80 = 1.2816  # cuantil normal para un rango del 80 %


def holt_winters(serie: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12,
                                    initialization_method="estimated").fit()


def mape(real: np.ndarray, estimado: np.ndarray) -> float:
    return float(np.mean(np.abs((real - estimado) / real)) * 100)


def analizar() -> dict:
    df = pd.read_csv(DATOS)
    serie = pd.Series(df["ventas_millones"].to_numpy(),
                      index=pd.PeriodIndex(df["mes"], freq="M").to_timestamp())

    entrenamiento, prueba = serie.iloc[:-PRUEBA], serie.iloc[-PRUEBA:]
    pred_hw = holt_winters(entrenamiento).forecast(PRUEBA).to_numpy()
    pred_ingenuo = entrenamiento.iloc[-12:].to_numpy()          # mismo mes del año anterior

    final = holt_winters(serie)
    proyeccion = final.forecast(HORIZONTE)
    sigma = float(np.std(final.resid, ddof=1))
    # El error se acumula con el horizonte: el rango se abre con la raíz de los pasos.
    margen = Z_80 * sigma * np.sqrt(np.arange(1, HORIZONTE + 1))

    return {
        "meses": [m.strftime("%Y-%m") for m in serie.index],
        "ventas": [round(float(v), 2) for v in serie],
        "prueba": {
            "meses": [m.strftime("%Y-%m") for m in prueba.index],
            "real": [round(float(v), 2) for v in prueba],
            "modelo": [round(float(v), 2) for v in pred_hw],
            "mape_modelo": round(mape(prueba.to_numpy(), pred_hw), 1),
            "mape_ingenuo": round(mape(prueba.to_numpy(), pred_ingenuo), 1),
        },
        "pronostico": {
            "meses": [m.strftime("%Y-%m") for m in proyeccion.index],
            "valor": [round(float(v), 2) for v in proyeccion],
            "bajo": [round(float(v - m), 2) for v, m in zip(proyeccion, margen)],
            "alto": [round(float(v + m), 2) for v, m in zip(proyeccion, margen)],
        },
        "total_6m": round(float(proyeccion.sum()), 1),
        "total_6m_anterior": round(float(serie.iloc[-12:-6].sum()), 1),
    }


if __name__ == "__main__":
    r = analizar()
    p = r["prueba"]
    print(f"[OK] Error en los 12 meses de prueba: modelo {p['mape_modelo']} % · regla simple {p['mape_ingenuo']} %")
    print(f"[OK] Próximos 6 meses: RD$ {r['total_6m']} M (mismos meses un año antes: {r['total_6m_anterior']} M)")
