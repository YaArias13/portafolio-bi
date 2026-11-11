"""
Genera el sitio del portafolio (docs/index.html) a partir de la plantilla y los datos.

Uso:
    python build.py

Qué editar:
    config.json              -> enlaces de contacto, repositorio y LinkedIn
    datos/proyectos.json     -> textos de cada proyecto
    datos/tableros.json      -> cifras ficticias que alimentan los tableros
    plantilla/index.html     -> diseño de la página
    plantilla/vendor/        -> Chart.js (biblioteca de gráficos)

GitHub Pages publica el contenido de la carpeta docs/.
"""
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PLANTILLA = RAIZ / "plantilla" / "index.html"
SALIDA = RAIZ / "docs" / "index.html"

CAMPOS_PROYECTO = {"id", "tipo", "titulo", "carpeta", "lista", "tags", "resumen",
                   "problema", "solucion", "resultado", "tablero"}


def leer_json(ruta: Path):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"ERROR: no existe {ruta.relative_to(RAIZ)}")
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: {ruta.relative_to(RAIZ)} no es JSON válido ({e})")


def validar_proyectos(proyectos: list) -> None:
    ids = [p.get("id") for p in proyectos]
    if len(ids) != len(set(ids)):
        sys.exit("ERROR: hay proyectos con el mismo id en datos/proyectos.json")
    for p in proyectos:
        faltan = CAMPOS_PROYECTO - p.keys()
        if faltan:
            sys.exit(f"ERROR: al proyecto {p.get('id', '?')} le faltan campos: {', '.join(sorted(faltan))}")


def js(valor) -> str:
    """Serializa a JSON seguro para incrustar dentro de una etiqueta <script>."""
    return json.dumps(valor, ensure_ascii=False).replace("</", "<\\/")


def construir() -> Path:
    config = leer_json(RAIZ / "config.json")
    proyectos = leer_json(RAIZ / "datos" / "proyectos.json")
    tableros = leer_json(RAIZ / "datos" / "tableros.json")
    validar_proyectos(proyectos)

    reemplazos = {
        "@@REPO@@": js(config.get("repo", "")),
        "@@LINKEDIN@@": js(config.get("linkedin", "")),
        "@@CORREO@@": config["correo"],
        "@@WA_NUMERO@@": config["whatsapp_numero"],
        "@@WA_VISIBLE@@": config["whatsapp_visible"],
        "@@DATOS@@": js(tableros),
        "@@PROYECTOS@@": js(proyectos),
    }

    html = PLANTILLA.read_text(encoding="utf-8")
    for marca, valor in reemplazos.items():
        if marca not in html:
            sys.exit(f"ERROR: la plantilla no contiene la marca {marca}")
        html = html.replace(marca, valor)
    if "@@" in html:
        sys.exit("ERROR: quedaron marcas @@...@@ sin reemplazar en la plantilla")

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(html, encoding="utf-8")
    (SALIDA.parent / ".nojekyll").write_text("", encoding="utf-8")
    # Chart.js va incluido en el repositorio para no depender de un CDN externo
    shutil.copytree(PLANTILLA.parent / "vendor", SALIDA.parent / "vendor", dirs_exist_ok=True)
    return SALIDA


if __name__ == "__main__":
    salida = construir()
    print(f"[OK] Sitio generado: {salida.relative_to(RAIZ)} ({salida.stat().st_size / 1024:.0f} KB)")
