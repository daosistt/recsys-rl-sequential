from src.config import DATA_DIR, np, pd, pq


def load_yambda_data(size: int = 1000) -> pd.DataFrame:
    """Загрузить первый батч пользовательских историй.

    Args:
        size: Максимальное число пользователей в батче.

    Returns:
        Таблица историй из файла listens.parquet в DATA_DIR.
    """
    with pq.ParquetFile(DATA_DIR / "listens.parquet") as pf:
        return next(pf.iter_batches(batch_size=size)).to_pandas()


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    *,
    train_end: int | None = None,
    val_end: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбить истории по общим временным границам, сохранив все признаки.

    Пользователи без train-истории пропускаются. Для оценки val/test
    предшествующую историю нужно отдельно добавить в контекст.

    Args:
        df: Таблица с одной строкой на пользователя и массивами взаимодействий,
            отсортированными по timestamp.
        train_size: Доля общего временного диапазона для train.
        val_size: Доля общего временного диапазона для validation.
        train_end: Исключающая правая граница train. Если обе границы не заданы,
            они вычисляются по train_size и val_size.
        val_end: Исключающая правая граница validation. Задаётся вместе
            с train_end; при явных границах доли не используются для разбиения.

    Returns:
        Три таблицы: train, validation и test, со всеми исходными колонками.
        Границы train_end и val_end записаны в attrs каждой таблицы.

    Raises:
        ValueError: Доли неположительны или их сумма не меньше единицы;
            задана только одна граница или train_end не меньше val_end.
    """
    if not (0 < train_size < 1 and 0 < val_size < 1 - train_size):
        raise ValueError("Доли должны быть положительными, а их сумма меньше 1")

    if train_end is None and val_end is None:
        start = min(int(ts[0]) for ts in df.timestamp if len(ts))
        stop = max(int(ts[-1]) for ts in df.timestamp if len(ts))
        
        train_end = start + int((stop - start) * train_size)
        val_end = start + int((stop - start) * (train_size + val_size))
    
    elif train_end is None or val_end is None or train_end >= val_end:
        raise ValueError("Укажите обе границы: train_end < val_end")

    parts = [[], [], []]
    for row in df.itertuples(index=False):

        a, b = np.searchsorted(row.timestamp, [train_end, val_end])

        if a == 0:
            continue
        for part, left, right in zip(parts, (0, a, b), (a, b, len(row.timestamp))):
            part.append({
                col: row.uid if col == "uid" else np.asarray(value)[left:right].copy()
                for col, value in row._asdict().items()
            })

    train, val, test = (pd.DataFrame(part, columns=df.columns) for part in parts)

    for frame in (train, val, test):
        frame.attrs.update(train_end=int(train_end), val_end=int(val_end))
    
    return train, val, test
