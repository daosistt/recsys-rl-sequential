# Отчёт

- `report.typ` — текст; раздел «Введение» оставлен автору.
- `report.pdf` — собранная версия.
- `assets/` — рисунки SVG и сведения об их источниках.
- `tables/` — числовые таблицы для проверки результатов.

Из корня проекта:

```bash
python -m report.build_figures
typst compile report/report.typ report/report.pdf
```

Первой команде нужны выгрузка `artifacts/report_exports`, исходные истории
трёх RL-запусков и checkpoint бейзлайна для его loss history.
Она только оформляет сохранённые данные: обучение и inference не запускает.
Для изменения текста достаточно второй команды; данные для неё уже в `report/`.
