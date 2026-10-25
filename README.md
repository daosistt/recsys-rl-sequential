# RL для музыкальных рекомендаций

GPT-2 на Yambda, reward model на CatBoost и RL-дообучение через REINFORCE.
Сравниваем награду, релевантность и поведение рекомендаций.

Установите зависимости из `requirements.txt` и откройте [experiment.ipynb](experiment.ipynb).
Данные (`listens.parquet`) положите в `data/`, сохранённые модели — в `artifacts/`.
В Colab/Kaggle пути настраиваются автоматически. По умолчанию повторное обучение выключено.

Код — в `src/`, результаты и выводы — в [отчёте](report/report.pdf).
