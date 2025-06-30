# LLM Compliance Toolkit

Un toolkit para evaluar la conformidad de los modelos de lenguaje (LLMs) con los principios de sistemas algorítmicos responsables definidos por la ACM.

## Objetivo

Este toolkit permite:
1. Recopilar metadatos de modelos LLM (información de creación, fecha, creador, datos de entrenamiento, etc.)
2. Almacenar pares de preguntas y respuestas obtenidos de varios LLMs
3. Evaluar la conformidad con los 9 principios ACM para sistemas algorítmicos responsables
4. Generar informes de cumplimiento con recomendaciones

## Principios Evaluados

1. Legitimidad y competencia
2. Minimización de daños
3. Seguridad y privacidad
4. Transparencia
5. Interpretabilidad y explicabilidad
6. Mantenibilidad
7. Contestabilidad y auditabilidad
8. Responsabilidad
9. Limitación de impactos ambientales

## Instalación

```bash
git clone [URL_DEL_REPOSITORIO]
cd llm_compliance_toolkit

pip install -r requirements.txt

python -m spacy download es_core_news_md
```

## Uso

### 1. Crear un Dataset de Ejemplo

Para generar un dataset de ejemplo con dos modelos (GPT-4 y Llama 2) y respuestas simuladas:

```bash
python examples/create_sample_dataset.py
```

### 2. Evaluar un Nuevo Modelo LLM

Para evaluar un nuevo modelo, sigue el asistente interactivo:

```bash
python examples/evaluate_model.py
```

Este script te guiará a través del proceso de:
- Registrar metadatos del modelo
- Hacer preguntas al modelo para cada principio ACM
- Registrar las respuestas
- Generar un informe de conformidad

### 3. Comparar Modelos

Para comparar la conformidad de varios modelos:

```bash
python examples/compare_models.py
```

Este script genera:
- Una tabla comparativa de niveles de conformidad
- Una visualización gráfica (heatmap)
- Un archivo JSON con los resultados de la comparación

## Estructura del Proyecto

- `data/`: Contiene datasets de ejemplo y evaluaciones
- `src/`: Código fuente del toolkit
  - `schema.py`: Define estructuras de datos
  - `toolkit.py`: Implementa el toolkit principal
- `examples/`: Scripts de ejemplo
  - `create_sample_dataset.py`: Crea un dataset de ejemplo
  - `evaluate_model.py`: Guía para evaluar un modelo
  - `compare_models.py`: Compara modelos evaluados
- `tests/`: Pruebas unitarias

## Extensiones Futuras

- Implementar evaluación automática usando LLMs para analizar respuestas
- Añadir soporte para modelos multilingües
- Crear una interfaz web
- Integrar con APIs de modelos populares
- Ampliar los criterios de evaluación

## Licencia

Este proyecto se distribuye bajo la licencia MIT. 