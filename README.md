# Portafolio BI · Yara Arias

Sitio estático del portafolio. La página se genera con Python a partir de una plantilla y archivos de datos, y se publica con GitHub Pages desde la carpeta `docs/`.

## Estructura

| Ruta | Contenido |
|---|---|
| `build.py` | Genera `docs/index.html` |
| `config.json` | Correo, WhatsApp, enlace del repositorio y de LinkedIn |
| `datos/proyectos.json` | Textos de cada proyecto |
| `datos/tableros.json` | Cifras ficticias que alimentan los tableros |
| `plantilla/index.html` | Diseño de la página |
| `docs/` | Sitio generado (lo que publica GitHub Pages) |

## Actualizar el sitio

```bash
python build.py
```

No requiere instalar nada: solo usa la biblioteca estándar de Python 3.9 o superior. Después de generar, sube los cambios y GitHub Pages actualiza la página.

Todos los datos son ficticios.
