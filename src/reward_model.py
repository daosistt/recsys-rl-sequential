from src.config import (
    CatBoostRegressor, Path, Ridge, gc, json, mean_absolute_error, mean_squared_error,
    np, pd, pickle, torch, tqdm,
)
from src.vocabulary import Vocabulary


def clear_memory() -> None:
    """Собрать недоступные объекты и освободить неиспользуемый CUDA-кеш."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_reward_examples(
    data: pd.DataFrame, context: pd.DataFrame | None, vocab: Vocabulary,
    seq_len: int, max_samples: int, seed: int = 42,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Выбрать взаимодействия с известным треком и предшествующей историей.

    Args:
        data: Истории оцениваемой части, включая played_ratio_pct.
        context: Истории предыдущих частей или None для train.
        vocab: Словарь бейзлайна.
        seq_len: Длина истории вместе с оцениваемым треком.
        max_samples: Максимальное число случайных примеров.
        seed: Seed для воспроизводимой выборки.

    Returns:
        Последовательности token_id, доли прослушивания и пары (строка, позиция события).
    """
    prefixes = {} if context is None else context.set_index("uid").item_id.to_dict()
    histories, rewards, positions, prefix_lengths = [], [], [], []
    for row in data.itertuples(index=False):
        prefix = list(prefixes.get(row.uid, []))[-(seq_len - 1):]
        tokens = vocab.encode(prefix + list(row.item_id))
        valid = np.flatnonzero(np.asarray(tokens[len(prefix):]) != vocab.unk_idx)
        if not prefix:
            valid = valid[valid > 0]  # Для первого события ещё нет истории.
        histories.append(tokens)
        rewards.append(np.asarray(row.played_ratio_pct, dtype=np.float32) / 100)
        positions.append(valid)
        prefix_lengths.append(len(prefix))

    # Равномерно выбираем события, не создавая отдельный объект на каждый listen.
    ends = np.cumsum([len(p) for p in positions])
    total = int(ends[-1])
    selected = np.random.default_rng(seed).choice(total, min(max_samples, total), replace=False)
    sequences, labels, samples = [], [], []
    for index in selected:
        user = int(np.searchsorted(ends, index, side="right"))
        offset = int(ends[user - 1]) if user else 0
        position = int(positions[user][index - offset])
        stop = prefix_lengths[user] + position + 1
        sequences.append(histories[user][max(0, stop - seq_len):stop])
        labels.append(rewards[user][position])
        samples.append((user, position))
    print(f"Выбрано {len(labels)} из {total} подходящих событий (всего {sum(map(len, data.item_id))})")
    return sequences, np.asarray(labels, dtype=np.float32), np.asarray(samples, dtype=np.int64)


def extract_reward_features(
    backbone: torch.nn.Module, sequences: list[list[int]], batch_size: int = 64,
) -> np.ndarray:
    """Получить замороженные признаки пары «история + трек».

    Args:
        backbone: Transformer из обученного GPT-2 в режиме eval.
        sequences: Последовательности, заканчивающиеся оцениваемым треком.
        batch_size: Размер батча.

    Returns:
        Массив признаков размера [число примеров, n_embd] на CPU.
    """
    backbone.eval()
    feature_device = next(backbone.parameters()).device
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Reward features"):
            batch = sequences[start:start + batch_size]
            length = max(map(len, batch))
            ids = torch.zeros((len(batch), length), dtype=torch.long, device=feature_device)
            for i, sequence in enumerate(batch):
                ids[i, -len(sequence):] = torch.tensor(sequence, device=feature_device)
            mask = (ids != 0).long()  # PAD=0 в нашем словаре.
            positions = (mask.cumsum(1) - 1).clamp(min=0)
            hidden = backbone(input_ids=ids, attention_mask=mask,
                              position_ids=positions, use_cache=False).last_hidden_state
            features.append(hidden[:, -1].cpu().numpy().copy())
    return np.concatenate(features)


def prepare_reward_features(
    model: torch.nn.Module,
    vocab: Vocabulary,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    params: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Заморозить GPT-2 и подготовить признаки и цели для трёх частей данных.

    Args:
        model: Обученный GPT2LMHeadModel.
        vocab: Его сохранённый словарь.
        train: Обучающие истории.
        val: Истории validation.
        test: Истории test.
        params: Размеры подвыборок, batch_size и seed.

    Returns:
        Словари признаков, целей и исходных item_id с ключами train, val и test.
    """
    model.requires_grad_(False)
    model.eval()
    test_context = train[["uid", "item_id"]].copy()
    test_context["item_id"] = [
        list(train_items) + list(val_items)
        for train_items, val_items in zip(train.item_id, val.item_id)
    ]
    features, targets, items = {}, {}, {}
    for name, data, context in (("train", train, None), ("val", val, train),
                                ("test", test, test_context)):
        print(f"Reward {name}:")
        sequences, labels, _ = make_reward_examples(
            data, context, vocab, model.config.n_positions,
            params[f"{name}_samples"], params["seed"],
        )
        features[name] = extract_reward_features(model.transformer, sequences, params["batch_size"])
        targets[name] = labels
        items[name] = np.asarray([vocab.token_to_item[sequence[-1]] for sequence in sequences])
        del sequences
        clear_memory()
    return features, targets, items


def _past_track_means(train: pd.DataFrame, items: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Вычислить средний отклик трека по train-событиям строго раньше запроса.

    Args:
        train: Обучающие истории с timestamp и played_ratio_pct.
        items: Объекты оцениваемых взаимодействий.
        times: Их временные метки.

    Returns:
        Средние отклики; при отсутствии трека — среднее по предшествующему train,
        а при полном отсутствии прошлого — нейтральное значение 0.5.
    """
    train_items = np.concatenate(train.item_id.to_numpy()).astype(np.int64)
    train_times = np.concatenate(train.timestamp.to_numpy()).astype(np.int64)
    rewards = np.concatenate(train.played_ratio_pct.to_numpy()) / 100

    order = np.argsort(train_times)
    counts = np.searchsorted(train_times[order], times, side="left")
    sums = np.r_[0, np.cumsum(rewards[order])]
    means = np.divide(sums[counts], counts, out=np.full(len(items), 0.5), where=counts > 0)

    order = np.lexsort((train_times, train_items))
    sorted_items, sorted_times, sorted_rewards = train_items[order], train_times[order], rewards[order]
    for item in np.unique(items):
        selected = items == item
        left = np.searchsorted(sorted_items, item, side="left")
        right = np.searchsorted(sorted_items, item, side="right")
        counts = np.searchsorted(sorted_times[left:right], times[selected], side="left")
        sums = np.r_[0, np.cumsum(sorted_rewards[left:right])]
        means[selected] = np.divide(sums[counts], counts, out=means[selected].copy(), where=counts > 0)
    return means


def make_reward_tabular(
    data: pd.DataFrame,
    context: pd.DataFrame | None,
    train: pd.DataFrame,
    samples: np.ndarray,
) -> np.ndarray:
    """Построить пять признаков без текущего отклика и будущих событий.

    Args:
        data: Истории оцениваемой части.
        context: Полные истории предыдущих частей или None для train.
        train: Источник статистик по трекам.
        samples: Пары (строка, позиция события) из make_reward_examples.

    Returns:
        Длительность трека, прошлые средние трека/пользователя, число повторов
        и доля прослушиваний менее 50%. Порядок задан REWARD_EXTRA_FEATURE_NAMES.
    """
    prefixes = {} if context is None else {row.uid: row for row in context.itertuples(index=False)}
    features = np.zeros((len(samples), 5), dtype=np.float64)
    items = np.zeros(len(samples), dtype=np.int64)
    times = np.zeros(len(samples), dtype=np.int64)
    for user, row in enumerate(data.itertuples(index=False)):
        selected = np.flatnonzero(samples[:, 0] == user)
        if len(selected) == 0:
            continue
        positions = samples[selected, 1]
        prefix = prefixes.get(row.uid)
        history_items = np.concatenate(([] if prefix is None else prefix.item_id, row.item_id))
        history_times = np.concatenate(([] if prefix is None else prefix.timestamp, row.timestamp))
        ratios = np.concatenate(([] if prefix is None else prefix.played_ratio_pct, row.played_ratio_pct)) / 100
        items[selected] = np.asarray(row.item_id)[positions]
        times[selected] = np.asarray(row.timestamp)[positions]
        past = np.searchsorted(history_times, times[selected], side="left")
        sums = np.r_[0, np.cumsum(ratios)]
        short_counts = np.r_[0, np.cumsum(ratios < 0.5)]

        features[selected, 0] = np.asarray(row.track_length_seconds)[positions]
        features[selected, 2] = np.divide(sums[past], past, out=np.full(len(past), 0.5), where=past > 0)
        features[selected, 3] = [
            np.count_nonzero(history_items[:count] == item)
            for count, item in zip(past, items[selected])
        ]
        features[selected, 4] = np.divide(short_counts[past], past, out=np.zeros(len(past)), where=past > 0)
    features[:, 1] = _past_track_means(train, items, times)
    return features.astype(np.float32)


def prepare_reward_tabular(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Vocabulary,
    seq_len: int,
    params: dict,
) -> dict[str, np.ndarray]:
    """Подготовить дополнительные признаки для тех же случайных примеров, что и GPT-2.

    Args:
        train: Обучающие истории.
        val: Истории validation.
        test: Истории test.
        vocab: Сохранённый словарь GPT-2.
        seq_len: Длина контекста GPT-2.
        params: Те же размеры подвыборок и seed, что в prepare_reward_features.

    Returns:
        Дополнительные признаки с ключами train, val и test.
    """
    columns = ["uid", "item_id", "timestamp", "played_ratio_pct"]
    test_context = train[columns].copy()
    for column in columns[1:]:
        test_context[column] = [np.concatenate((a, b)) for a, b in zip(train[column], val[column])]
    features = {}
    for name, data, context in (("train", train, None), ("val", val, train), ("test", test, test_context)):
        _, _, samples = make_reward_examples(
            data, context, vocab, seq_len, params[f"{name}_samples"], params["seed"],
        )
        features[name] = make_reward_tabular(data, context, train, samples)
        del samples
        clear_memory()
    return features


def fit_reward_head(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> Ridge:
    """Обучить Ridge на train и выбрать регуляризацию по validation MAE.

    Args:
        features: Признаки по частям данных.
        targets: Доли прослушивания по частям данных.
        alphas: Кандидаты на коэффициент L2-регуляризации.

    Returns:
        Линейная модель с минимальным validation MAE.
    """
    heads, scores = [], []
    for alpha in alphas:
        head = Ridge(alpha=alpha, solver="svd").fit(features["train"], targets["train"])
        prediction = head.predict(features["val"])
        heads.append(head)
        scores.append(mean_absolute_error(targets["val"], prediction))
    best = heads[int(np.argmin(scores))]
    print(f"Выбрана alpha={best.alpha}, validation MAE={min(scores) * 100:.3f} п.п.")
    return best


def evaluate_reward_head(
    head: Ridge,
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    train: pd.DataFrame,
    items: dict[str, np.ndarray],
    extra_heads: dict[str, Ridge | CatBoostRegressor] | None = None,
    extra_features: dict[str, dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Сравнить reward model, общее среднее и среднее по треку на validation/test.

    Args:
        head: Обученная линейная модель.
        features: Признаки по частям данных.
        targets: Доли прослушивания по частям данных.
        train: Полные train-истории для расчёта средних откликов по трекам.
        items: Исходные item_id тех же примеров, что в features и targets.
        extra_heads: Дополнительные модели на тех же признаках, с названиями для таблицы.
        extra_features: Отдельные признаки для дополнительных моделей, если они отличаются.

    Returns:
        Таблица ошибок в процентных пунктах, с индексами Split и Model.
    """
    train_mean = float(targets["train"].mean())
    train_events = pd.DataFrame({
        "item_id": np.concatenate(train.item_id.to_numpy()),
        "reward": np.concatenate(train.played_ratio_pct.to_numpy()) / 100,
    })
    track_means = train_events.groupby("item_id")["reward"].mean()
    fallback_mean = float(train_events.reward.mean())
    rows = []
    for split in ("val", "test"):
        actual = targets[split]
        predictions = {
            "Train mean": np.full(len(actual), train_mean),
            "Track mean": pd.Series(items[split]).map(track_means).fillna(fallback_mean).to_numpy(),
            "GPT-2 + Ridge": head.predict(features[split]),
        }
        for name, extra_head in (extra_heads or {}).items():
            head_features = (extra_features or {}).get(name, features)
            predictions[name] = extra_head.predict(head_features[split])
        for name, predicted in predictions.items():
            rows.append({
                "Split": split, "Model": name,
                "MAE (п.п.)": mean_absolute_error(actual, predicted) * 100,
                "RMSE (п.п.)": np.sqrt(mean_squared_error(actual, predicted)) * 100,
            })
    return pd.DataFrame(rows).set_index(["Split", "Model"])


def fit_reward_catboost(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    params: dict,
) -> CatBoostRegressor:
    """Обучить CatBoost на признаках GPT-2 с early stopping по validation.

    Args:
        features: Те же признаки train/val, которые использовались для Ridge.
        targets: Доли прослушивания; test при обучении не используется.
        params: Параметры CatBoost, включая early_stopping_rounds.

    Returns:
        Модель с лучшим количеством деревьев по validation MAE.
    """
    head = CatBoostRegressor(**params)
    head.fit(
        features["train"], targets["train"],
        eval_set=(features["val"], targets["val"]),
        use_best_model=True,
    )
    print(f"CatBoost: сохранено {head.tree_count_} деревьев.")
    return head


def save_reward_model(
    head: Ridge | CatBoostRegressor,
    metrics: pd.DataFrame,
    output_dir: Path,
    baseline_dir: Path,
    baseline_config: dict,
    params: dict,
    train_targets: np.ndarray,
    feature_names: list[str] | None = None,
) -> None:
    """Сохранить модель отклика, метрики и параметры эксперимента.

    Args:
        head: Обученная модель отклика на признаках GPT-2.
        metrics: Таблица MAE/RMSE.
        output_dir: Папка результатов reward model.
        baseline_dir: Папка исходного GPT-2; его веса и словарь не копируются.
        baseline_config: Сохранённый конфиг исходного GPT-2.
        params: Параметры подвыборок и извлечения признаков.
        train_targets: Обучающие цели для сохранения среднего отклика.
        feature_names: Названия признаков в порядке подачи модели.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "reward_head.pkl", "wb") as f:
        pickle.dump(head, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "backbone_run": baseline_dir.name,
            "sampling": params,
            "split": baseline_config["split"],
            "head_type": type(head).__name__,
            "head_params": head.get_params(),
            "feature_names": feature_names,
            "target": "played_ratio_pct / 100, unclipped",
            "candidate_filter": "target item must be in baseline vocabulary",
            "train_mean": float(train_targets.mean()),
        }, f, indent=2)
    metrics.to_csv(output_dir / "metrics.csv")
    print(f"Модель, конфиг и метрики сохранены в {output_dir}")
