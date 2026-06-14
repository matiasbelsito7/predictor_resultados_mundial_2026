# Predictor Mundial 2026

Pipeline completo para la predicción de resultados del Mundial FIFA 2026 mediante scraping de datos, ingeniería de características, construcción de un índice de fuerza y simulación Monte Carlo del torneo.

---

# 1. Recolección de Datos (Scraping)

El proyecto comienza con un proceso de **scraping automático sobre Flashscore**, diseñado para obtener información detallada de las 48 selecciones clasificadas al Mundial 2026.

Para cada selección se recopilan:

* Resultados históricos.
* Goles convertidos y recibidos.
* Estadísticas ofensivas y defensivas.
* Métricas avanzadas (xG, xGA, tiros, posesión, etc.).
* Información del plantel y ratings individuales.
* Contexto competitivo de cada encuentro.

Los datos se almacenan en archivos estructurados para permitir actualizaciones periódicas sin necesidad de repetir todo el proceso de extracción.

---

# 2. Ingeniería de Features

La ingeniería de características constituye el núcleo del proyecto. El objetivo fue transformar estadísticas crudas en variables capaces de representar de manera más fiel el nivel competitivo actual de cada selección.

## 2.1 Ventana de rendimiento reciente

En lugar de utilizar todo el historial disponible, se decidió trabajar principalmente con los **últimos 20 partidos disputados por cada selección**.

La hipótesis detrás de esta decisión es que:

* El rendimiento reciente es más representativo que el histórico.
* Permite capturar cambios generacionales.
* Refleja mejor el impacto de nuevos entrenadores.
* Reduce la influencia de resultados demasiado antiguos.

---

## 2.2 Ponderación temporal

No todos los partidos dentro de esos 20 encuentros reciben el mismo peso.

Se implementó un esquema de **decaimiento temporal**, donde los encuentros más recientes tienen una influencia significativamente mayor que los más antiguos.

### Ejemplo conceptual

* Partido disputado hace 2 meses → peso alto.
* Partido disputado hace 1 año → peso intermedio.
* Partido disputado hace varios años → peso bajo.

Este mecanismo permite que la forma actual de una selección tenga mayor impacto sobre las métricas finales.

---

## 2.3 Ponderación por importancia del partido

El contexto competitivo de un encuentro influye directamente sobre su peso dentro del modelo.

Se asignaron ponderaciones diferenciadas según el torneo:

### Competiciones consideradas

* Copa del Mundo.
* Eliminatorias mundialistas.
* Eurocopa.
* Copa América.
* Copa Africana de Naciones.
* Copa Asiática.
* Concacaf Gold Cup.
* UEFA Nations League.
* Amistosos internacionales.

### Criterio

Los torneos oficiales reciben una ponderación superior a los amistosos debido a que:

* Los equipos suelen utilizar sus mejores planteles.
* Existe una mayor exigencia táctica.
* Los resultados tienen consecuencias reales.

Por el contrario, los amistosos suelen incluir pruebas tácticas, rotaciones y menor intensidad competitiva.

---

## 2.4 Ponderación por calidad del rival

Para contextualizar correctamente cada resultado se incorporó la fortaleza del adversario mediante el **Ranking FIFA**.

### Ejemplos

* Una victoria frente a una selección Top 10 aporta más información que una victoria frente a una selección ubicada fuera del Top 100.
* Una derrota ante una potencia mundial es menos penalizada que una derrota ante una selección de bajo ranking.

Esta ponderación afecta tanto las métricas ofensivas como las defensivas.

---

## 2.5 Ponderación por confederación

Además del ranking FIFA, se consideró la región competitiva a la que pertenece cada rival.

### Confederaciones contempladas

* CONMEBOL
* UEFA
* CAF
* AFC
* CONCACAF
* OFC

Históricamente existen diferencias de nivel entre regiones, por lo que los resultados obtenidos frente a determinadas confederaciones pueden tener un significado distinto.

Por ejemplo:

* Enfrentar regularmente a selecciones UEFA o CONMEBOL suele implicar una oposición promedio más exigente.
* Algunas regiones presentan una mayor dispersión de nivel entre sus integrantes.

La ponderación busca capturar estas diferencias estructurales.

---

## 2.6 Construcción de Variables

A partir de los partidos ponderados se generó un amplio conjunto de variables agrupadas en distintas categorías.

### Variables ofensivas

Miden la capacidad de generar peligro:

* Goles por partido.
* xG (Expected Goals).
* Remates totales.
* Remates al arco.
* Conversión de ocasiones.
* Producción ofensiva reciente.
* Producción ofensiva ajustada por rival.

### Variables defensivas

Miden la solidez sin posesión:

* Goles recibidos.
* xGA (Expected Goals Against).
* Diferencia de gol.
* Vallas invictas.
* Eficiencia defensiva.
* Capacidad para limitar ocasiones rivales.

### Variables de control del juego

Representan el dominio global de los encuentros:

* Posesión.
* Precisión de pases.
* Control territorial.
* Volumen ofensivo.
* Intensidad competitiva.

### Variables de consistencia

Buscan medir la estabilidad del rendimiento:

* Desviación estándar del xG.
* Variabilidad de resultados.
* Regularidad ofensiva.
* Regularidad defensiva.

Un equipo consistente es considerado más confiable que otro con rendimientos extremadamente variables.

### Variables de calidad de plantilla

Derivadas de los ratings de los futbolistas:

* Rating promedio del once titular.
* Rating promedio de los mejores jugadores.
* Profundidad del plantel.
* Equilibrio entre líneas.
* Dispersión de ratings.

---

## 2.7 Índice de Fuerza

Una vez generadas todas las variables:

1. Se normalizan para volverlas comparables.
2. Se corrigen variables donde valores altos representan peor rendimiento.
3. Se aplican penalizaciones por datos faltantes (NaN).
4. Se combinan mediante un sistema de pesos.

El esquema final prioriza:

* Rendimiento ofensivo.
* Rendimiento defensivo.
* Calidad de plantilla.
* Consistencia.
* Control del juego.

El resultado es un **índice de fuerza normalizado entre 0 y 1**, que resume el nivel estimado de cada selección.

Este índice constituye la principal entrada del simulador.

---

# 3. Modelo de Predicción de Partidos

Cada encuentro se modela utilizando la diferencia entre los índices de fuerza de ambos equipos.

A partir de dicha diferencia se calculan probabilidades de:

* Victoria del Equipo A.
* Empate.
* Victoria del Equipo B.

Para ello se utiliza una función tipo **softmax**, que transforma las diferencias de fuerza en probabilidades coherentes.

Posteriormente se generan marcadores mediante distribuciones de **Poisson**, una de las técnicas más utilizadas en modelado estadístico de fútbol.

En rondas eliminatorias:

* Los empates se resuelven mediante una simulación de penales.
* El equipo más fuerte recibe una ligera ventaja probabilística.

---

# 4. Simulación del Mundial

El torneo se reproduce respetando la estructura oficial del Mundial FIFA 2026.

## Fase de grupos

* 12 grupos de 4 selecciones.
* Todos los partidos son simulados individualmente.
* Se calculan posiciones, diferencia de gol y criterios de desempate.

## Clasificación

Avanzan:

* Los dos primeros de cada grupo.
* Los ocho mejores terceros.

## Eliminatorias

* Round of 32 (32 equipos).
* Round of 16.
* Cuartos de final.
* Semifinales.
* Final.

El cuadro eliminatorio sigue el bracket oficial del torneo.

---

# 5. Simulación Monte Carlo

Para estimar probabilidades realistas se utiliza el método de **Monte Carlo**.

El torneo completo se simula miles de veces.

## En cada simulación

1. Se juega la fase de grupos.
2. Se determinan los clasificados.
3. Se construye el cuadro eliminatorio.
4. Se disputa todo el torneo.
5. Se obtiene un campeón.

## Resultados calculados

Tras miles de repeticiones se estiman probabilidades de:

* Clasificar al Round of 32.
* Alcanzar Round of 16.
* Llegar a cuartos de final.
* Alcanzar semifinales.
* Llegar a la final.
* Ganar la Copa del Mundo.

De esta manera el sistema no produce únicamente una predicción puntual, sino una distribución completa de probabilidades que refleja la incertidumbre inherente al fútbol.
