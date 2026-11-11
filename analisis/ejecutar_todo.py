"""
Corre los tres análisis y guarda los resultados que muestra el sitio.

Uso (desde la raíz del repositorio):
    pip install -r analisis/requirements.txt
    python analisis/ejecutar_todo.py
    python build.py
"""
import importlib.util
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parent / "datos" / "ciencia.json"


def cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> None:
    cargar("01_generar_datos").main()
    resultados = {
        "fuga": cargar("02_modelo_fuga").analizar(),
        "pronostico": cargar("03_pronostico").analizar(),
        "segmentos": cargar("04_segmentacion").analizar(),
    }
    SALIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Resultados guardados en {SALIDA.relative_to(AQUI.parent)}")


if __name__ == "__main__":
    main()
