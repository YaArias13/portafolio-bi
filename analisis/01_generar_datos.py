"""
Genera los datos simulados que usan los tres análisis.

Nada aquí viene de una empresa real: son clientes y ventas inventados, con
relaciones puestas a propósito para que los métodos tengan algo que encontrar.

Salidas (carpeta analisis/datos_simulados/):
    clientes.csv         6,000 clientes con su comportamiento y si se fueron o no
    ventas_mensuales.csv 48 meses de ventas con tendencia y estacionalidad
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEMILLA = 2026
CARPETA = Path(__file__).resolve().parent / "datos_simulados"

# Cuatro tipos de cliente. Cada fila: proporción y promedio de cada variable.
PERFILES = {
    #                    prop  antig  prod  saldo   digital  recencia
    "digital_joven":    (0.30,   14,  2.2,  38_000,   26,      6),
    "tradicional":      (0.34,   96,  2.8,  95_000,    3,     24),
    "alto_valor":       (0.14,   78,  4.6, 520_000,   15,      8),
    "poco_activo":      (0.22,   30,  1.2,   9_000,    2,     85),
}


def generar_clientes(n: int, rng: np.random.Generator) -> pd.DataFrame:
    nombres = list(PERFILES)
    proporciones = [PERFILES[k][0] for k in nombres]
    perfil = rng.choice(nombres, size=n, p=proporciones)
    df = pd.DataFrame({"cliente_id": np.arange(1, n + 1), "perfil_simulado": perfil})

    medias = pd.DataFrame({k: PERFILES[k][1:] for k in nombres},
                          index=["antig", "prod", "saldo", "digital", "recencia"]).T.loc[perfil]
    df["antiguedad_meses"] = np.clip(rng.gamma(4, medias["antig"].to_numpy() / 4), 1, 300).round()
    df["productos_activos"] = np.clip(rng.poisson(medias["prod"].to_numpy()), 1, 8)
    df["saldo_promedio"] = (rng.lognormal(np.log(medias["saldo"].to_numpy()), 0.55)).round(-2)
    df["transacciones_digitales_mes"] = rng.poisson(medias["digital"].to_numpy())
    df["dias_sin_actividad"] = np.clip(rng.gamma(2, medias["recencia"].to_numpy() / 2), 0, 365).round()
    df["reclamos_12m"] = rng.poisson(0.35, n)
    df["tiene_banca_en_linea"] = (df["transacciones_digitales_mes"] > 0).astype(int)
    df["segmento"] = np.where(df["saldo_promedio"] > 300_000, "Premium",
                     np.where(rng.random(n) < 0.18, "Pyme", "Masivo"))

    # Probabilidad de fuga: sube con la inactividad y los reclamos, baja con la
    # vinculación (productos), el uso digital y la antigüedad. Más ruido.
    z = (-1.75
         + 0.016 * df["dias_sin_actividad"]
         + 0.55 * df["reclamos_12m"]
         - 0.42 * df["productos_activos"]
         - 0.035 * df["transacciones_digitales_mes"]
         - 0.006 * df["antiguedad_meses"]
         - 0.25 * np.log10(df["saldo_promedio"] / 10_000)
         + rng.normal(0, 0.9, n))
    df["se_fue"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def generar_ventas(rng: np.random.Generator) -> pd.DataFrame:
    meses = pd.period_range("2022-11", periods=48, freq="M")
    t = np.arange(48)
    tendencia = 3.2 + 0.045 * t                                  # millones de RD$
    estacional = np.array([0.10, 0.55, -0.35, -0.30, 0.05, -0.05,   # nov, dic, ene...
                           0.20, 0.00, -0.10, -0.15, 0.00, 0.05])[t % 12]
    ventas = tendencia + estacional * (1 + 0.01 * t) + rng.normal(0, 0.12, 48)
    return pd.DataFrame({"mes": meses.astype(str), "ventas_millones": ventas.round(3)})


def main() -> None:
    rng = np.random.default_rng(SEMILLA)
    CARPETA.mkdir(exist_ok=True)
    clientes = generar_clientes(6_000, rng)
    ventas = generar_ventas(rng)
    clientes.to_csv(CARPETA / "clientes.csv", index=False)
    ventas.to_csv(CARPETA / "ventas_mensuales.csv", index=False)
    print(f"[OK] {len(clientes):,} clientes · tasa de fuga {clientes['se_fue'].mean():.1%}")
    print(f"[OK] {len(ventas)} meses de ventas ({ventas['mes'].iloc[0]} a {ventas['mes'].iloc[-1]})")


if __name__ == "__main__":
    main()
