# Paso a Paso Detallado: ongoing-bps-state-short-term

## 📋 Descripción General

Este proyecto calcula el **estado actual (partial state)** de procesos de negocio en curso a partir de un log de eventos histórico y un modelo BPMN. Además, puede ejecutar **simulaciones de corto plazo (short-term simulation)** usando ese estado parcial como punto de partida, permitiendo predecir el comportamiento futuro del proceso desde el estado actual.

---

## 🎯 Objetivo Principal

Dado un log de eventos hasta un punto de corte temporal (`start_time`), el sistema:
1. **Identifica casos en curso** (ongoing cases)
2. **Calcula el estado de control de flujo** de cada caso usando N-Gram Index
3. **Identifica actividades en curso, habilitadas, gateways habilitados y eventos habilitados**
4. **Opcionalmente**: Ejecuta una simulación de corto plazo desde ese estado parcial hasta un horizonte temporal (`simulation_horizon`)

---

## 🏗️ Arquitectura del Sistema

```
[Log CSV] + [BPMN] + [JSON Params]
    ↓
[InputHandler] → Lee y procesa entradas
    ↓
[EventLogProcessor] → Filtra eventos hasta cut-off, calcula enabled_time
    ↓
[BPMNHandler] → Construye N-Gram Index y Reachability Graph
    ↓
[StateComputer] → Calcula estado de cada caso
    ↓
[output.json] → Estado parcial del proceso
    ↓
[Prosimos Short-Term Simulation] → (Opcional) Simula desde estado parcial
```

---

## 📁 Estructura del Proyecto

```
ongoing-bps-state-short-term/
├── main.py                    # Punto de entrada CLI
├── src/
│   ├── runner.py              # Orquestador principal
│   ├── input_handler.py       # Manejo de entradas (log, BPMN, params)
│   ├── event_log_processor.py # Procesamiento del log de eventos
│   ├── bpmn_handler.py        # Manejo del modelo BPMN
│   ├── state_computer.py      # Cálculo del estado de casos
│   ├── process_state_prosimos_run.py  # Integración con Prosimos
│   └── misc.py                # Utilidades
├── test/                      # Scripts de evaluación
├── samples/                   # Datos de ejemplo
└── outputs/                   # Resultados generados
```

---

## 🔄 FLUJO COMPLETO PASO A PASO

### FASE 1: PREPARACIÓN DE ENTRADAS

#### Paso 1.1: Preparar Log de Eventos (CSV)
- **Formato requerido**: CSV con columnas estándar o mapeables
- **Columnas estándar**:
  - `CaseId`: Identificador del caso
  - `Activity`: Nombre de la actividad
  - `Resource`: Recurso que ejecuta la actividad
  - `StartTime`: Timestamp de inicio (ISO-8601)
  - `EndTime`: Timestamp de fin (ISO-8601, puede ser NULL para eventos en curso)
  - `enabled_time`: (Opcional) Timestamp cuando la actividad fue habilitada
- **Formato de timestamps**: ISO-8601 con fracciones de segundo
  - Ejemplo: `2025-01-20T09:36:26.463000+00:00`
  - Si falta `.000`, se añade automáticamente
- **Eventos en curso**: Tienen `EndTime = NULL` o `EndTime > start_time`

#### Paso 1.2: Preparar Modelo BPMN
- **Formato**: Archivo BPMN 2.0 (XML)
- **Contenido requerido**:
  - Tareas (tasks)
  - Gateways (exclusive, parallel, inclusive, etc.)
  - Eventos (start, intermediate, end)
  - Sequence flows (flujos de secuencia)
- **Nota**: El sistema divide internamente cada tarea en `+START` y `+COMPLETE` para el análisis

#### Paso 1.3: Preparar Parámetros de Simulación (JSON)
- **Formato**: JSON con parámetros estocásticos para Prosimos
- **Contenido típico**:
  ```json
  {
    "arrival_time_distribution": {...},
    "arrival_time_calendar": [...],
    "resource_profiles": [...],
    "task_resource_distribution": [...],
    "gateway_branching_probabilities": [...]
  }
  ```

#### Paso 1.4: Configurar Mapeo de Columnas (Opcional)
- Si el CSV usa nombres de columnas diferentes, proporcionar mapeo JSON:
  ```json
  {
    "CaseId": "case_id",
    "Activity": "task_name",
    "Resource": "user",
    "StartTime": "start_timestamp",
    "EndTime": "end_timestamp"
  }
  ```

---

### FASE 2: PROCESAMIENTO INICIAL (`runner.py` → `InputHandler`)

#### Paso 2.1: Crear InputHandler
- **Función**: `InputHandler.__init__(args)`
- **Proceso**:
  1. **Parsear mapeo de columnas**:
     - Si se proporciona `column_mapping`, parsea JSON
     - Si no, usa mapeo por defecto (columnas ya con nombres estándar)
  2. **Leer log de eventos**:
     - Lee CSV usando `pandas.read_csv()`
     - Aplica mapeo de columnas para renombrar
     - Valida que existan columnas requeridas
     - Convierte `StartTime` y `EndTime` a `pd.Timestamp` (UTC)
     - Asegura fracciones de segundo (añade `.000` si falta)
  3. **Crear EventLogIDs**:
     - Instancia `EventLogIDs` con nombres estándar
  4. **Leer modelo BPMN**:
     - Usa `ongoing_process_state.utils.read_bpmn_model()`
  5. **Parsear parámetros JSON**:
     - Lee y parsea el archivo JSON de parámetros

#### Paso 2.2: Procesar Log de Eventos (`EventLogProcessor`)
- **Función**: `EventLogProcessor.process()`
- **Proceso**:
  1. **Filtrar casos en curso** (si hay `start_time`):
     - Un caso es "en curso" si:
       - Tiene al menos un evento con `StartTime > start_time`, O
       - Tiene eventos con `EndTime = NULL`, O
       - Tiene eventos con `EndTime > start_time`
     - Filtra el DataFrame para mantener solo casos en curso
  2. **Filtrar eventos hasta cut-off**:
     - Mantiene solo eventos con `StartTime <= start_time`
  3. **Marcar eventos en curso**:
     - Si `EndTime = NULL` o `EndTime > start_time`, establece `EndTime = NaT`
  4. **Calcular `enabled_time`** (si no existe):
     - **Temporalmente**: Rellena `EndTime` faltantes con `max(EndTime) + 1 hora`
     - **Crea Concurrency Oracle**:
       - Usa `OverlappingConcurrencyOracle` de `pix-framework`
       - Configura con `ConcurrencyThresholds(df=0.5)`
     - **Calcula enabled_time**:
       - Llama `concurrency_oracle.add_enabled_times(df)`
       - Determina cuándo cada actividad fue habilitada basándose en concurrencia
     - **Revierte relleno temporal**: Restaura `EndTime = NaT` donde estaba originalmente
  5. **Retorna DataFrame procesado** y `concurrency_oracle` para uso posterior

---

### FASE 3: CONSTRUCCIÓN DE ÍNDICES BPMN (`BPMNHandler`)

#### Paso 3.1: Parsear BPMN XML
- **Función**: `BPMNHandler.parse_bpmn_xml()`
- **Proceso**:
  1. **Extraer actividades (tasks)**:
     - Busca elementos `<bpmn:task>`
     - Crea mapeo: `{task_id: task_name}`
     - Crea mapeo inverso: `{task_name: task_id}`
  2. **Extraer eventos**:
     - Busca: `startEvent`, `intermediateCatchEvent`, `intermediateThrowEvent`, `endEvent`
     - Crea mapeo: `{event_id: event_name}`
     - Identifica `end_events` (para detectar casos finalizados)
  3. **Extraer gateways**:
     - Busca: `exclusiveGateway`, `parallelGateway`, `inclusiveGateway`, etc.
     - Crea mapeo: `{gateway_id: gateway_type}`
  4. **Extraer sequence flows**:
     - Busca `<bpmn:sequenceFlow>`
     - Crea mapeo: `{flow_id: target_ref}`
     - Crea mapeo inverso: `{flow_id: source_ref}`

#### Paso 3.2: Construir Modelo BPMN Extendido
- **Función**: `compute_extended_bpmn_model()`
- **Propósito**: Divide cada tarea en dos nodos: `+START` y `+COMPLETE`
- **Proceso**:
  1. **Para cada tarea**:
     - Crea nodo `{task_id}+START`
     - Crea nodo `{task_id}+COMPLETE`
     - Crea flujo interno: `task_id → {task_id}+START → {task_id}+COMPLETE`
  2. **Para eventos**:
     - Se añaden sin dividir (a menos que `treat_event_as_task=True`)
  3. **Para gateways**:
     - Se añaden sin dividir
  4. **Actualizar flujos**:
     - Si un flujo apunta a una tarea dividida, apunta a `{task_id}+START`
     - Si un flujo sale de una tarea dividida, sale de `{task_id}+COMPLETE`

#### Paso 3.3: Construir Reachability Graph
- **Función**: `BPMNHandler.build_n_gram_index()`
- **Proceso**:
  1. **Obtener grafo de alcanzabilidad**:
     - Usa `extended_bpmn_model.get_reachability_graph()`
     - Representa todos los estados alcanzables del proceso
  2. **Construir N-Gram Index**:
     - Crea `NGramIndex(reachability_graph, n_gram_size_limit=20)`
     - `NGramIndex` indexa secuencias de actividades (n-gramas) a estados del proceso
     - Llama `n_gram_index.build()` para construir el índice
  3. **Retorna**:
     - `n_gram_index`: Para buscar estados basados en secuencias
     - `reachability_graph`: Para análisis de alcanzabilidad

---

### FASE 4: CÁLCULO DEL ESTADO DE CASOS (`StateComputer`)

#### Paso 4.1: Agrupar Eventos por Caso
- **Función**: `StateComputer.compute_case_states()`
- **Proceso inicial**:
  1. Agrupa el log procesado por `CaseId`
  2. Ordena eventos de cada caso por `StartTime`

#### Paso 4.2: Identificar Actividades en Curso (Ongoing Activities)
- **Para cada caso**:
  1. **Filtrar actividades en curso**:
     - Busca eventos con `EndTime = NaT` (NULL)
  2. **Para cada actividad en curso**:
     - Obtiene `task_id` usando `bpmn_handler.get_task_id_by_name(activity_name)`
     - Extrae: `start_time`, `resource`, `enabled_time`
     - Si `enabled_time` no existe, intenta calcularlo usando `concurrency_oracle`
  3. **Construye lista**:
     ```python
     ongoing_activities = [{
         "id": task_id,
         "label": activity_name,
         "start_time": start_time,
         "resource": resource,
         "enabled_time": enabled_time
     }]
     ```

#### Paso 4.3: Construir N-Gram del Caso
- **Proceso**:
  1. **Crear secuencia de eventos**:
     - Para cada evento en el caso:
       - Si tiene `start_time`, añade `{activity}+START` a la secuencia
       - Si tiene `end_time`, añade `{activity}+COMPLETE` a la secuencia
     - Ordena eventos por timestamp
  2. **Construir n-gram**:
     - Prefija con `NGramIndex.TRACE_START`
     - Ejemplo: `[TRACE_START, "TaskA+START", "TaskA+COMPLETE", "TaskB+START", ...]`
  3. **Buscar estado en N-Gram Index**:
     - Llama `n_gram_index.get_best_marking_state_for(n_gram)`
     - Retorna un "marking" (marcado) que representa el estado del proceso
     - El marking contiene IDs de sequence flows y actividades con tokens

#### Paso 4.4: Extraer Estado de Control de Flujo
- **Del marking obtenido**:
  1. **Separar flows y actividades**:
     - `state_flows`: IDs de sequence flows con tokens
     - `state_activities`: IDs de actividades con tokens
  2. **Reparar actividades "fantasma"**:
     - Si una actividad tiene token en el marking pero NO está en `ongoing_activities`:
       - Es un artefacto del N-Gram Index
       - Se reemplaza por sus incoming flows (flujos entrantes)
       - Esto corrige discrepancias entre el estado inferido y el estado real del log
  3. **Resultado**:
     ```python
     control_flow_state = {
         "flows": [flow_id1, flow_id2, ...],
         "activities": [activity_id1, activity_id2, ...]
     }
     ```

#### Paso 4.5: Calcular Actividades Habilitadas (Enabled Activities)
- **Proceso**:
  1. **Para cada flow con token** (`state_flows`):
     - Obtiene `target_ref` del flow (hacia dónde apunta)
     - Si `target_ref` es una actividad:
       - Calcula `enabled_time`:
         - Si no hay actividades finalizadas: usa `min(StartTime)` del caso
         - Si hay actividades finalizadas:
           - Usa `concurrency_oracle.enabled_since()` con un evento temporal
           - El evento temporal tiene `start_time = max(EndTime) + 1 segundo`
       - Añade a `enabled_activities`:
         ```python
         {
             "id": target_ref,
             "enabled_time": enabled_time
         }
         ```

#### Paso 4.6: Calcular Gateways Habilitados (Enabled Gateways)
- **Proceso**:
  1. **Para cada flow con token** (`state_flows`):
     - Obtiene `target_ref` del flow
     - Si `target_ref` es un gateway exclusivo (`exclusiveGateway`):
       - **Verifica upstream tasks**:
         - Obtiene todas las tareas upstream del gateway (recorriendo hacia atrás)
         - Si alguna tarea upstream tiene token en `state_activities` (está en curso):
           - **Omite este gateway** (no está realmente habilitado aún)
       - Si no hay tareas upstream en curso:
         - Calcula `enabled_time`:
           - Si hay actividades finalizadas: usa `max(EndTime)` de tareas upstream
           - Si no: usa `min(StartTime)` del caso
         - Añade a `enabled_gateways`:
           ```python
           {
               "id": gateway_id,
               "enabled_time": enabled_time
           }
           ```
  2. **Filtro especial**:
     - Si algún gateway habilitado es un `endEvent`, **se excluye el caso completo**
     - (El caso está finalizado, no debe estar en el estado parcial)

#### Paso 4.7: Calcular Eventos Habilitados (Enabled Events)
- **Proceso**:
  1. **Para cada flow con token** (`state_flows`):
     - Obtiene `target_ref` del flow
     - Si `target_ref` es un evento (no end event):
       - Calcula `enabled_time` similar a actividades habilitadas
       - Añade a `enabled_events`:
         ```python
         {
             "id": event_id,
             "enabled_time": enabled_time
         }
         ```

#### Paso 4.8: Construir Estado Completo del Caso
- **Para cada caso**, se construye:
  ```python
  case_states[case_id] = {
      "control_flow_state": {
          "flows": [flow_id1, flow_id2, ...],
          "activities": [activity_id1, activity_id2, ...]
      },
      "ongoing_activities": [
          {
              "id": task_id,
              "label": activity_name,
              "start_time": timestamp,
              "resource": resource_name,
              "enabled_time": timestamp
          },
          ...
      ],
      "enabled_activities": [
          {
              "id": activity_id,
              "enabled_time": timestamp
          },
          ...
      ],
      "enabled_gateways": [
          {
              "id": gateway_id,
              "enabled_time": timestamp
          },
          ...
      ],
      "enabled_events": [
          {
              "id": event_id,
              "enabled_time": timestamp
          },
          ...
      ]
  }
  ```

---

### FASE 5: GENERACIÓN DE OUTPUT (`output.json`)

#### Paso 5.1: Calcular Última Llegada de Caso
- **Proceso**:
  - Agrupa por caso y obtiene `min(StartTime)` de cada caso
  - `last_case_arrival = max(min(StartTime) por caso)`
  - Representa el timestamp del inicio del caso más reciente

#### Paso 5.2: Construir Estructura de Salida
- **Estructura**:
  ```json
  {
      "last_case_arrival": "2025-01-20T09:14:12.710000+00:00",
      "cases": {
          "case_id_1": {
              "control_flow_state": {...},
              "ongoing_activities": [...],
              "enabled_activities": [...],
              "enabled_gateways": [...],
              "enabled_events": [...]
          },
          "case_id_2": {...},
          ...
      }
  }
  ```

#### Paso 5.3: Guardar `output.json`
- **Función**: `json.dump(output_data, f, default=str, indent=4)`
- **Ubicación**: `./output.json` (directorio actual)

---

### FASE 6: SIMULACIÓN DE CORTO PLAZO (Opcional)

#### Paso 6.1: Verificar Flag de Simulación
- **Condición**: `if simulate == True`
- Si no se activa, el proceso termina aquí

#### Paso 6.2: Preparar Parámetros de Simulación
- **Función**: `run_short_term_simulation()`
- **Parámetros**:
  - `start_date`: Timestamp de inicio (opcional, por defecto `last_case_arrival`)
  - `total_cases`: Número de casos a simular (por defecto 20)
  - `bpmn_model`: Ruta al archivo BPMN
  - `json_sim_params`: Ruta al JSON de parámetros
  - `process_state`: El diccionario `output_data` (estado parcial)
  - `simulation_horizon`: Timestamp límite de la simulación (ISO-8601)

#### Paso 6.3: Ejecutar Prosimos con Estado Parcial
- **Función**: `prosimos.simulation_engine.run_simulation()`
- **Proceso**:
  1. **Carga estado parcial**:
     - Convierte strings de timestamps a `datetime` en `process_state`
     - Prosimos usa este estado como punto de partida
  2. **Inicializa simulación**:
     - Carga modelo BPMN
     - Carga parámetros estocásticos (JSON)
     - **Restaura casos en curso**:
       - Para cada caso en `process_state["cases"]`:
         - Restaura actividades en curso (con sus recursos y tiempos)
         - Restaura tokens en sequence flows y actividades
         - Marca actividades habilitadas con sus `enabled_time`
  3. **Ejecuta simulación**:
     - Continúa desde el estado parcial
     - Genera nuevos casos según `arrival_time_distribution`
     - Simula hasta `simulation_horizon` (no hasta `total_cases`)
     - Si no hay `simulation_horizon`, simula `total_cases` casos
  4. **Genera salidas**:
     - `sim_log_csv`: Log de eventos simulados (CSV)
     - `sim_stats_csv`: Estadísticas de simulación (CSV)

#### Paso 6.4: Resultados de Simulación
- **Archivos generados**:
  - `simulation_log.csv`: Eventos simulados desde el estado parcial
  - `simulation_stats.csv`: Métricas de simulación (cycle_time, processing_time, etc.)

---

## 🔧 MÓDULOS DETALLADOS

### `src/runner.py`
- **`run_process_state_and_simulation()`**: Función principal que orquesta todo el proceso
  - Paso A: Construye objeto `args` para `InputHandler`
  - Paso B: Usa `InputHandler` para leer entradas
  - Paso C: Procesa log de eventos
  - Paso D: Construye N-Gram Index
  - Paso E: Calcula estado de casos
  - Paso F: Escribe `output.json`
  - Paso G: (Opcional) Ejecuta simulación de corto plazo

### `src/input_handler.py`
- **`InputHandler`**: Maneja lectura y parseo de entradas
  - `read_event_log()`: Lee CSV, aplica mapeo, valida, parsea timestamps
  - `read_bpmn_model()`: Lee modelo BPMN usando `ongoing_process_state`
  - `parse_bpmn_parameters()`: Lee JSON de parámetros
  - `parse_column_mapping()`: Parsea mapeo de columnas JSON
  - `get_event_log_ids()`: Crea instancia `EventLogIDs`
  - `ensure_fractional_seconds()`: Añade `.000` a timestamps sin fracciones

### `src/event_log_processor.py`
- **`EventLogProcessor`**: Procesa el log según punto de corte
  - `process()`:
    - Filtra casos en curso
    - Filtra eventos hasta `start_time`
    - Marca eventos en curso (`EndTime = NaT`)
    - Calcula `enabled_time` usando `OverlappingConcurrencyOracle`

### `src/bpmn_handler.py`
- **`BPMNHandler`**: Maneja operaciones con BPMN
  - `parse_bpmn_xml()`: Extrae elementos del XML (tasks, events, gateways, flows)
  - `build_n_gram_index()`: Construye N-Gram Index y Reachability Graph
  - `get_task_id_by_name()`: Mapea nombre de actividad a ID
  - `get_node_type()`: Determina tipo de elemento BPMN
  - `get_incoming_flows()`: Obtiene flows entrantes de una actividad
  - `get_upstream_tasks_through_gateways()`: Obtiene tareas upstream de un gateway
  - `is_end_event()`: Verifica si un elemento es end event
  - `compute_extended_bpmn_model()`: Divide tareas en +START/+COMPLETE

### `src/state_computer.py`
- **`StateComputer`**: Calcula estado de cada caso
  - `compute_case_states()`: Función principal
    - Agrupa por caso
    - Identifica actividades en curso
    - Construye n-gram del caso
    - Busca estado en N-Gram Index
    - Calcula actividades/gateways/eventos habilitados
  - `_compute_gateway_enabled_time()`: Calcula tiempo de habilitación de gateway

### `src/process_state_prosimos_run.py`
- **`run_short_term_simulation()`**: Wrapper para Prosimos con estado parcial
  - Prepara parámetros
  - Llama `prosimos.simulation_engine.run_simulation()` con `process_state` y `simulation_horizon`
- **`run_basic_simulation()`**: Simulación estándar sin estado parcial
- **`parse_datetime()`**: Parsea timestamps ISO-8601 (monkey-patch para Prosimos)

---

## 📊 FORMATO DE `output.json`

```json
{
    "last_case_arrival": "2025-01-20T09:14:12.710000+00:00",
    "cases": {
        "1718": {
            "control_flow_state": {
                "flows": ["flow_id_1", "flow_id_2"],
                "activities": ["activity_id_1", "activity_id_2"]
            },
            "ongoing_activities": [
                {
                    "id": "task_id",
                    "label": "Check credit history",
                    "start_time": "2025-01-20T09:36:26.463000+00:00",
                    "resource": "Clerk-000002",
                    "enabled_time": "2025-01-20T09:36:05.192000+00:00"
                }
            ],
            "enabled_activities": [
                {
                    "id": "activity_id",
                    "enabled_time": "2025-01-20T09:40:00.000000+00:00"
                }
            ],
            "enabled_gateways": [
                {
                    "id": "gateway_id",
                    "enabled_time": "2025-01-20T09:45:00.000000+00:00"
                }
            ],
            "enabled_events": [
                {
                    "id": "event_id",
                    "enabled_time": "2025-01-20T09:50:00.000000+00:00"
                }
            ]
        }
    }
}
```

---

## 🚀 USO DEL SISTEMA

### Ejemplo Básico (Solo Estado)

```bash
python main.py \
    event_log.csv \
    model.bpmn \
    parameters.json \
    --start_time "2025-01-20T09:00:00+00:00"
```

### Ejemplo con Simulación

```bash
python main.py \
    event_log.csv \
    model.bpmn \
    parameters.json \
    --start_time "2025-01-20T09:00:00+00:00" \
    --simulate \
    --simulation_horizon "2025-01-20T12:00:00+00:00" \
    --total_cases 50 \
    --sim_log_csv "simulation_log.csv" \
    --sim_stats_csv "simulation_stats.csv"
```

### Ejemplo con Mapeo de Columnas

```bash
python main.py \
    event_log.csv \
    model.bpmn \
    parameters.json \
    --column_mapping '{"CaseId":"case_id","Activity":"task","Resource":"user","StartTime":"start","EndTime":"end"}'
```

---

## 🔍 CONCEPTOS CLAVE

### N-Gram Index
- **Propósito**: Indexa secuencias de actividades a estados del proceso
- **Funcionamiento**:
  - Construye un índice de n-gramas (secuencias de n actividades)
  - Dada una secuencia de actividades, busca el estado más probable en el Reachability Graph
  - Permite inferir el estado de control de flujo desde el historial de eventos

### Reachability Graph
- **Propósito**: Representa todos los estados alcanzables del proceso
- **Nodos**: Estados del proceso (marcados con tokens)
- **Aristas**: Transiciones entre estados (ejecución de actividades)

### Concurrency Oracle
- **Propósito**: Determina cuándo una actividad fue habilitada
- **Método**: `OverlappingConcurrencyOracle`
  - Analiza solapamiento temporal de actividades
  - Si dos actividades se solapan, pueden ejecutarse concurrentemente
  - Calcula `enabled_time` basándose en dependencias temporales

### Estado Parcial (Partial State)
- **Definición**: Estado del proceso en un momento específico (cut-off)
- **Contiene**:
  - Tokens en sequence flows
  - Actividades en curso (con recursos y tiempos)
  - Actividades/gateways/eventos habilitados (con tiempos de habilitación)

### Simulación de Corto Plazo
- **Propósito**: Predecir comportamiento futuro desde el estado actual
- **Características**:
  - Usa estado parcial como punto de partida
  - Simula hasta un horizonte temporal (no número fijo de casos)
  - Restaura casos en curso con sus recursos y tiempos reales

---

## 📝 NOTAS IMPORTANTES

1. **Timestamps**: Deben estar en UTC y con formato ISO-8601
2. **Eventos en curso**: Deben tener `EndTime = NULL` o `EndTime > start_time`
3. **N-Gram Index**: Requiere modelo BPMN válido con estructura completa
4. **Prosimos**: Requiere rama especial `short-term-simulation` para soportar estado parcial
5. **Concurrency Oracle**: Usa umbral de solapamiento del 50% por defecto
6. **Gateways**: Solo se consideran habilitados si no hay tareas upstream en curso

---

## 🎯 CASOS DE USO

1. **Monitoreo en Tiempo Real**: Calcular estado actual de procesos en ejecución
2. **Predicción de Corto Plazo**: Simular qué pasará en las próximas horas/días
3. **Análisis de Casos en Curso**: Identificar qué actividades están activas y cuáles están habilitadas
4. **Optimización de Recursos**: Usar estado parcial para planificar asignación de recursos
5. **Evaluación de Cambios**: Comparar estados antes/después de cambios en el proceso

---

## 🔗 DEPENDENCIAS PRINCIPALES

- **ongoing-process-state**: Para N-Gram Index y manejo de BPMN
- **pix-framework**: Para Concurrency Oracle y procesamiento de logs
- **Prosimos** (rama short-term-simulation): Para simulación con estado parcial
- **pandas**: Para manipulación de DataFrames
- **networkx**: Para análisis de grafos (usado internamente)

---

## 📈 FLUJO DE DATOS COMPLETO

```
[Log CSV] → [InputHandler] → [EventLogProcessor] → [Log Procesado]
                                                          ↓
[BPMN XML] → [BPMNHandler] → [N-Gram Index] ────────────┘
                                                          ↓
[JSON Params] → [InputHandler] ───────────────────────────┘
                                                          ↓
[StateComputer] → [Estado de Casos] → [output.json]
                                                          ↓
[Prosimos] ← [Estado Parcial] ← [output.json] (si --simulate)
                                                          ↓
[Simulation Log + Stats]
```

---

Este documento describe el funcionamiento completo del sistema `ongoing-bps-state-short-term` para calcular estados parciales de procesos y ejecutar simulaciones de corto plazo.

