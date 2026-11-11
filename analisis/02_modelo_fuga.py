"""
Modelo predictivo de fuga de clientes.

Pregunta: con lo que sabemos de un cliente hoy, ¿qué tan probable es que se vaya?

Compara tres opciones sobre los mismos clientes de prueba:
    1. Un puntaje de reglas hecho a mano (como el del proyecto de churn en SQL)
    2. Regresión logística
    3. Gradient boosting

Devuelve un diccionario con las métricas y las series para los gráficos.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 2026
DATOS = Path(__file__).resolve().parent / "datos_simulados" / "clientes.csv"

VARIABLES = {
    "dias_sin_actividad": "Días sin actividad",
    "productos_activos": "Productos activos",
    "reclamos_12m": "Reclamos en 12 meses",
    "transacciones_digitales_mes": "Uso digital al mes",
    "antiguedad_meses": "Antigüedad",
    "saldo_promedio": "Saldo promedio",
    "tiene_banca_en_linea": "Tiene banca en línea",
}


def puntaje_reglas(df: pd.DataFrame) -> pd.Series:
    """Puntos fijos por regla, definidos a mano."""
    return (25 * (df["productos_activos"] <= 1)
            + 25 * (df["dias_sin_actividad"] > 60)
            + 20 * (df["transacciones_digitales_mes"] < 3)
            + 15 * (df["reclamos_12m"] >= 1)
            + 10 * (df["antiguedad_meses"] < 12)
            + 5 * (df["tiene_banca_en_linea"] == 0))


def tabla_deciles(y: np.ndarray, prob: np.ndarray) -> list[dict]:
    """Ordena de mayor a menor riesgo y parte en 10 grupos iguales."""
    orden = pd.DataFrame({"y": y, "p": prob}).sort_values("p", ascending=False).reset_index(drop=True)
    orden["decil"] = pd.qcut(orden.index, 10, labels=False) + 1
    base = orden["y"].mean()
    g = orden.groupby("decil")["y"].agg(["mean", "sum", "count"])
    g["captura_acum"] = g["sum"].cumsum() / orden["y"].sum()
    return [{"decil": int(d), "tasa": round(100 * r["mean"], 1), "lift": round(r["mean"] / base, 2),
             "captura_acum": round(100 * r["captura_acum"], 1)} for d, r in g.iterrows()]


def puntos_roc(y: np.ndarray, prob: np.ndarray, n: int = 41) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(y, prob)
    malla = np.linspace(0, 1, n)
    return [[round(float(x), 3), round(float(np.interp(x, fpr, tpr)), 3)] for x in malla]


def analizar() -> dict:
    df = pd.read_csv(DATOS)
    X, y = df[list(VARIABLES)], df["se_fue"]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)

    logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_ent, y_ent)
    boosting = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=200,
                                              random_state=SEMILLA).fit(X_ent, y_ent)
    prob = {
        "Reglas a mano": puntaje_reglas(X_pru).to_numpy() / 100,
        "Regresión logística": logistica.predict_proba(X_pru)[:, 1],
        "Gradient boosting": boosting.predict_proba(X_pru)[:, 1],
    }
    comparacion = [{"modelo": m, "auc": round(roc_auc_score(y_pru, p), 3),
                    "precision_media": round(average_precision_score(y_pru, p), 3),
                    "captura_top20": tabla_deciles(y_pru.to_numpy(), p)[1]["captura_acum"]}
                   for m, p in prob.items()]
    mejor = max(comparacion[1:], key=lambda r: r["auc"])["modelo"]
    modelo = logistica if mejor == "Regresión logística" else boosting

    imp = permutation_importance(modelo, X_pru, y_pru, scoring="roc_auc", n_repeats=15, random_state=SEMILLA)
    importancia = sorted(([VARIABLES[c], round(float(v), 4)] for c, v in zip(X.columns, imp.importances_mean)),
                         key=lambda r: -r[1])

    return {
        "clientes": int(len(df)), "entrenamiento": int(len(X_ent)), "prueba": int(len(X_pru)),
        "tasa_fuga": round(100 * float(y.mean()), 1),
        "comparacion": comparacion, "mejor": mejor,
        "roc": {m: puntos_roc(y_pru.to_numpy(), p) for m, p in prob.items()},
        "deciles": tabla_deciles(y_pru.to_numpy(), prob[mejor]),
        "importancia": importancia,
    }


if __name__ == "__main__":
    r = analizar()
    print(f"[OK] {r['prueba']:,} clientes de prueba · fuga {r['tasa_fuga']} %")
    for c in r["comparacion"]:
        print(f"     {c['modelo']:<22} AUC {c['auc']:.3f} · captura en el 20 % superior: {c['captura_top20']} %")
