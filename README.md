# Requirement Chatbot

Proyecto base para extraer entidades, normalizar texto y preparar un flujo híbrido con reglas y fallback.

## Archivos

- `main.py`: punto de entrada por consola
- `quick_test.py`: prueba rápida del flujo
- `requirement_chatbot.py`: orquestación principal
- `hybrid_extractor.py`: combina extractores
- `rule_entity_extractor.py`: extracción por reglas
- `llm_fallback_extractor.py`: espacio para integración con LLM
- `calculation_engine.py`: cálculos derivados

## Uso

```bash
python main.py
```

O ejecuta una prueba rápida:

```bash
python quick_test.py
```
