# Презентация и отчет

Презентация и теховский файл лежат в `report/presentation/`.
Отчет по экспериментам находится в `report/exp_report/`

# Скрипты и запуск

## RegBench

| Путь к скрипту | Назначение |
| --- | --- |
| `scripts/env.sh` | Общие настройки окружения. |
| `scripts/run_regbench.sh` | Обучение выбранного метода внимания на DFA-задаче RegBench. |
| `scripts/launch_regbench.sh` | Точка входа для обучения на выбранном GPU в фоне. |
| `scripts/launch_eval_regbench.sh` | Оценка лучшего сохранённого чекпоинта. |
| `scripts/run_analysis_regbench.sh` | Сравнение softmax и Epanechnikov / sparsemax по сохранённым результатам. |
| `scripts/run_analysis_variants.sh` | Расширенное сравнение после оценки всех методов и анализа двух базовых методов. |
| `scripts/run_paired_bootstrap.sh` | Парный bootstrap для softmax и Epanechnikov на одинаковых тестовых примерах. |
| `scripts/build_figures.py` | Построение графиков. |

## BabiStories

| Путь к скрипту | Назначение |
| --- | --- |
| `scripts/download_babistories.sh` | Загрузка частей архива BabiStories с проверкой контрольных сумм. |
| `scripts/prepare_babistories.sh` | Токенизация предварительно распакованного датасета. |
| `scripts/run_lm_smoke.sh` | Один шаг обучения и оценки. |
| `upstream/Library/train_memory_mosaics.py` | Точка входа для полноценного обучения. |

## Дополнительные проверки

| Проверка | Путь к скрипту |
| --- | --- |
| Соответствие ядер и механизмов внимания | `scripts/verify_theory.py` |
| Выбор разных методов внимания в модели | `scripts/smoke_models.py` |
| Воспроизводимость генерации DFA | `scripts/run_check_dfa_determinism.sh` |
