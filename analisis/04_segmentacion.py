"""
Segmentación de clientes.

Pregunta: ¿qué grupos naturales hay entre los clientes, sin decirle al algoritmo cuáles buscar?

Pasos:
    1. Se escalan las variables (el saldo va en logaritmo porque es muy asimétrico).
    2. Se prueba K-means con 2 a 7 grupos y se elige el número con mejor silueta.
    3. Se describe cada grupo con sus promedios y se le pone un nombre.
    4. Se proyecta a dos dimensiones (PCA) solo para poder dibujarlo.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 2026
DATOS = Path(__file__).resolve().parent / "datos_simulados" / "clientes.csv"
VARIABLES = ["antiguedad_meses", "productos_activos", "saldo_promedio",
             "transacciones_digitales_mes", "dias_sin_actividad"]


def nombrar(perfil: pd.Series, general: pd.Series) -> str:
    """Nombre según lo que más distingue al grupo del promedio general."""
    if perfil["saldo_promedio"] > 3 * general["saldo_promedio"]:
        return "Alto valor"
    if perfil["dias_sin_actividad"] > 1.6 * general["dias_sin_actividad"]:
        return "Poco activos"
    if perfil["transacciones_digitales_mes"] > 1.4 * general["transacciones_digitales_mes"]:
        return "Digitales"
    if perfil["antiguedad_meses"] > 1.3 * general["antiguedad_meses"]:
        return "Tradicionales"
    return "Intermedios"


def analizar() -> dict:
    df = pd.read_csv(DATOS)
    X = df[VARIABLES].copy()
    X["saldo_promedio"] = np.log10(X["saldo_promedio"])
    Z = StandardScaler().fit_transform(X)

    rng = np.random.default_rng(SEMILLA)
    muestra = rng.choice(len(Z), 2500, replace=False)            # la silueta es costosa
    siluetas = {}
    for k in range(2, 8):
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Z)
        siluetas[k] = round(float(silhouette_score(Z[muestra], etiquetas[muestra])), 3)
    k = max(siluetas, key=siluetas.get)
    df["grupo"] = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Z)

    general = df[VARIABLES].mean()
    perfiles = df.groupby("grupo")[VARIABLES + ["se_fue"]].mean()
    perfiles["clientes"] = df["grupo"].value_counts()
    perfiles["nombre"] = [nombrar(p, general) for _, p in perfiles.iterrows()]
    if perfiles["nombre"].duplicated().any():
        raise SystemExit("Dos grupos recibieron el mismo nombre: revisar las reglas de nombrar().")
    perfiles = perfiles.sort_values("clientes", ascending=False).reset_index()
    orden = {g: i for i, g in enumerate(perfiles["grupo"])}

    pca = PCA(n_components=2, random_state=SEMILLA)
    coords = pca.fit_transform(Z)
    puntos = rng.choice(len(Z), 700, replace=False)

    return {
        "clientes": int(len(df)), "k": int(k),
        "siluetas": [[int(a), b] for a, b in siluetas.items()],
        "varianza_2d": round(100 * float(pca.explained_variance_ratio_.sum()), 1),
        "grupos": [{
            "nombre": r["nombre"], "clientes": int(r["clientes"]),
            "pct": round(100 * r["clientes"] / len(df), 1),
            "antiguedad": round(float(r["antiguedad_meses"])),
            "productos": round(float(r["productos_activos"]), 1),
            "saldo": round(float(r["saldo_promedio"]), -2),
            "digital": round(float(r["transacciones_digitales_mes"]), 1),
            "inactividad": round(float(r["dias_sin_actividad"])),
            "fuga": round(100 * float(r["se_fue"]), 1),
        } for _, r in perfiles.iterrows()],
        "puntos": [[round(float(coords[i, 0]), 2), round(float(coords[i, 1]), 2), orden[int(df["grupo"].iat[i])]]
                   for i in puntos],
    }


if __name__ == "__main__":
    r = analizar()
    print(f"[OK] Mejor número de grupos: {r['k']} (siluetas: {dict(map(tuple, r['siluetas']))})")
    for g in r["grupos"]:
        print(f"     {g['nombre']:<14} {g['pct']:>5} % · saldo {g['saldo']:>9,.0f} · digital {g['digital']:>4} · fuga {g['fuga']} %")
