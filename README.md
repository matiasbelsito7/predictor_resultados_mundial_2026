# predictor_resultados_mundial_2026
Pipeline de predicción del Mundial 2026: scrapeo automático de estadísticas de selecciones desde Flashscore, limpieza y consolidación de datos, ingeniería de variables ofensivas, defensivas y de rendimiento, cálculo de un índice de fuerza ponderado y simulación completa del torneo mediante Monte Carlo para estimar probabilidades de avance y campeon
1. Recolección de datos (Scraping)
El proyecto comienza con un proceso de scraping automático sobre Flashscore para obtener información detallada de las 48 selecciones clasificadas al Mundial 2026. Para cada equipo se descargan estadísticas históricas de partidos, rendimiento reciente, resultados, goles, métricas avanzadas y datos de plantel.
Los datos se almacenan de forma estructurada para permitir actualizaciones periódicas sin necesidad de repetir todo el proceso. Esto facilita mantener el modelo alineado con el estado actual de cada selección.
