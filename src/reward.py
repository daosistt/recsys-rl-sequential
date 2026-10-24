from src.config import CatBoostRegressor, GPT2LMHeadModel, np, pd
from src.reward_model import extract_reward_features
from src.vocabulary import Vocabulary


def prepare_track_statistics(train: pd.DataFrame) -> dict:
    """Подготовить статистики треков один раз для последующего расчёта наград.

    Args:
        train: Только train-истории с item_id, timestamp, played_ratio_pct
            и track_length_seconds.

    Returns:
        Длительности, популярность и временные ряды накопленного отклика.
        Популярность — число train-прослушиваний / максимум по трекам.
        Средние отклики будут вычисляться на момент каждого запроса.
    """
    items = np.concatenate(train.item_id.to_numpy()).astype(np.int64)
    times = np.concatenate(train.timestamp.to_numpy()).astype(np.int64)
    ratios = np.concatenate(train.played_ratio_pct.to_numpy()) / 100
    lengths = np.concatenate(train.track_length_seconds.to_numpy())

    time_order = np.argsort(times)
    statistics = {
        "times": times[time_order],
        "sums": np.r_[0, np.cumsum(ratios[time_order])],
        "tracks": {}, "lengths": {}, "popularity": {},
    }
    order = np.lexsort((times, items))
    sorted_times, sorted_ratios = times[order], ratios[order]
    track_ids, starts, counts = np.unique(items[order], return_index=True, return_counts=True)
    max_count = counts.max()
    for item, start, count in zip(track_ids, starts, counts):
        stop = start + count
        statistics["tracks"][int(item)] = (
            sorted_times[start:stop], np.r_[0, np.cumsum(sorted_ratios[start:stop])],
        )
        statistics["lengths"][int(item)] = float(lengths[order[start]])
        statistics["popularity"][int(item)] = float(count / max_count)
    return statistics


def make_candidate_features(
    histories: list[dict],
    items: list[int],
    timestamps: list[int],
    statistics: dict,
) -> np.ndarray:
    """Собрать пять признаков для произвольных треков без их текущих откликов.

    Args:
        histories: История для каждого кандидата: item_id, timestamp,
            played_ratio_pct. Для статистик нужна вся доступная история.
        items: Исходные item_id кандидатов из словаря, по одному на историю.
        timestamps: Время запроса для каждой пары «история + кандидат».
        statistics: Результат prepare_track_statistics, построенный на train.

    Returns:
        Матрица признаков в порядке REWARD_EXTRA_FEATURE_NAMES. Используются
        только события строго раньше времени запроса. Отклики неизвестных
        сгенерированных взаимодействий нельзя добавлять в histories.
    """
    features = np.zeros((len(items), 5), dtype=np.float32)
    for i, (history, item, timestamp) in enumerate(zip(histories, items, timestamps)):
        global_count = np.searchsorted(statistics["times"], timestamp, side="left")
        track_mean = statistics["sums"][global_count] / global_count if global_count else 0.5
        track_times, track_sums = statistics["tracks"][item]
        track_count = np.searchsorted(track_times, timestamp, side="left")
        if track_count:
            track_mean = track_sums[track_count] / track_count

        past = np.asarray(history["timestamp"]) < timestamp
        past_items = np.asarray(history["item_id"])[past]
        ratios = np.asarray(history["played_ratio_pct"])[past] / 100
        features[i] = [
            statistics["lengths"][item],
            track_mean,
            float(ratios.mean()) if len(ratios) else 0.5,
            np.count_nonzero(past_items == item),
            float((ratios < 0.5).mean()) if len(ratios) else 0.0,
        ]
    return features


def predict_listening_ratio(
    model: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    histories: list[list[int]],
    items: list[int],
    tabular_features: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Предсказать отклик CatBoost по замороженным эмбеддингам и пяти признакам.

    Args:
        model: Отдельная копия исходного GPT-2 для reward model, не RL-политика.
        head: Сохранённый CatBoost из reward_catboost_stats_minfreq5.
        vocab: Словарь исходного GPT-2.
        histories: Предыдущие последовательности исходных item_id.
        items: Кандидаты из словаря; PAD и UNK сюда не передаются.
        tabular_features: Пять признаков из make_candidate_features для тех же пар.
        batch_size: Размер батча извлечения эмбеддингов.

    Returns:
        Предсказанные доли прослушивания без обрезания диапазона и без градиентов.
    """
    model.requires_grad_(False)
    sequences = [
        vocab.encode((list(history) + [item])[-model.config.n_positions:])
        for history, item in zip(histories, items)
    ]
    embeddings = extract_reward_features(model.transformer, sequences, batch_size)
    features = np.column_stack((embeddings, tabular_features))
    return head.predict(features)


def listening_reward(predicted_ratios: np.ndarray) -> np.ndarray:
    """Использовать прогноз доли прослушивания как награду.

    Args:
        predicted_ratios: Прогнозы CatBoost в долях, не в процентах.

    Returns:
        Награды без штрафов и обрезания диапазона.
    """
    return np.asarray(predicted_ratios)


def repeat_penalty_reward(
    predicted_ratios: np.ndarray,
    histories: list[list[int]],
    items: list[int],
    penalty: float = 0.1,
) -> np.ndarray:
    """Вычесть фиксированный штраф за выбор трека, уже встречавшегося в истории.

    Args:
        predicted_ratios: Прогнозы CatBoost в долях.
        histories: Полные истории item_id; можно включить ранее сгенерированные
            треки, чтобы также штрафовать повторы внутри списка.
        items: Кандидаты, по одному на историю.
        penalty: Размер штрафа: 0.1 соответствует 10 процентным пунктам.

    Returns:
        Прогноз минус penalty, если кандидат встречается в истории хотя бы раз.
    """
    repeats = np.asarray([item in history for history, item in zip(histories, items)])
    return listening_reward(predicted_ratios) - penalty * repeats


def popularity_penalty_reward(
    predicted_ratios: np.ndarray,
    items: list[int],
    statistics: dict,
    penalty: float = 0.1,
) -> np.ndarray:
    """Уменьшить награду пропорционально популярности трека в train.

    Args:
        predicted_ratios: Прогнозы CatBoost в долях.
        items: Исходные item_id кандидатов из словаря.
        statistics: Результат prepare_track_statistics. Популярность фиксируется
            по всему train и не использует validation/test.
        penalty: Максимальный штраф, применяемый к самому популярному треку.

    Returns:
        Прогноз минус penalty * (частота трека / максимальная train-частота).
    """
    popularity = np.asarray([statistics["popularity"][item] for item in items])
    return listening_reward(predicted_ratios) - penalty * popularity
