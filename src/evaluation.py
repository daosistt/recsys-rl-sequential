from src.config import CatBoostRegressor, Counter, GPT2Config, GPT2LMHeadModel, Path, json, np, pd, pickle, torch, tqdm
from src.reinforce import policy_log_probs, score_actions
from src.vocabulary import Vocabulary


def load_model(
    artifact_dir: Path, device: torch.device
) -> tuple[GPT2LMHeadModel, Vocabulary, dict]:
    """Загрузить лучший чекпоинт, его словарь и параметры эксперимента.

    Args:
        artifact_dir: Папка запуска с vocab.pkl и checkpoints/best.pt.
        device: Устройство для модели.

    Returns:
        Модель в режиме eval, сохранённый словарь и конфиг эксперимента.
    """
    with open(artifact_dir / "experiment_config.json") as f:
        saved_config = json.load(f)
    with open(artifact_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    checkpoint = torch.load(
        artifact_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
    )
    model = GPT2LMHeadModel(GPT2Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, saved_config


def prepare_evaluation_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Vocabulary,
    seq_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Подготовить по одному next-item примеру на пользователя.

    Args:
        train: Обучающие истории.
        val: Истории validation, добавляемые в контекст.
        test: Истории test; первый объект служит целью.
        vocab: Сохранённый словарь модели.
        seq_len: Максимальная длина контекста.

    Returns:
        Контексты в token_id и цели в исходных item_id, включая неизвестные.
    """
    histories = {row.uid: list(row.item_id) for row in train.itertuples(index=False)}
    for row in val.itertuples(index=False):
        histories[row.uid].extend(row.item_id)

    contexts, targets = [], []
    for row in test.itertuples(index=False):
        if len(row.item_id) == 0:
            continue
        contexts.append(vocab.encode(histories[row.uid][-seq_len:]))
        targets.append([int(row.item_id[0])])
    return contexts, targets


def recommend_top_k(
    model: GPT2LMHeadModel,
    contexts: list[list[int]],
    vocab: Vocabulary,
    k: int = 20,
    batch_size: int = 8,
) -> list[list[int]]:
    """Получить топ-K треков для каждого контекста по всему каталогу.

    Args:
        model: Обученная модель; оценка выполняется на её устройстве.
        contexts: Непустые истории в token_id длиной не больше n_positions.
        vocab: Словарь модели.
        k: Число рекомендаций.
        batch_size: Размер батча для оценки.

    Returns:
        Списки исходных item_id без PAD и UNK. Знакомые треки допустимы.
    """
    model.eval()
    device = next(model.parameters()).device
    seq_len = model.config.n_positions
    k = min(k, len(vocab) - 2)
    recommendations = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(contexts), batch_size), desc="Evaluation"):
            batch_contexts = contexts[start:start + batch_size]
            input_ids = torch.full(
                (len(batch_contexts), seq_len), vocab.pad_idx,
                dtype=torch.long, device=device,
            )
            for i, context in enumerate(batch_contexts):
                input_ids[i, -len(context):] = torch.tensor(context, device=device)

            # Padding слева; позиции считаются только по реальным токенам.
            attention_mask = (input_ids != vocab.pad_idx).long()
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
            scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                logits_to_keep=1,
                use_cache=False,
            ).logits[:, -1, :]
            scores[:, [vocab.pad_idx, vocab.unk_idx]] = -float("inf")
            top_tokens = scores.topk(k, dim=-1).indices.cpu().tolist()
            recommendations.extend(vocab.decode(tokens) for tokens in top_tokens)
    return recommendations


def evaluate_metrics(
    recommendations: list[list[int]],
    targets: list[list[int]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Усреднить метрики по примерам, считая неизвестные цели промахами.

    Args:
        recommendations: Уникальные item_id по убыванию score для каждого примера.
        targets: Непустые списки целевых item_id в том же порядке.
        ks: Значения K; рекомендации должны содержать не меньше max(ks) объектов,
            кроме случая, когда весь каталог меньше K.

    Returns:
        Таблица Recall@K, NDCG@K и MAP@K с индексом K.
    """
    results = []
    for k in ks:
        results.append({
            "K": k,
            "Recall@K": float(np.mean([
                recall_at_k(predicted, actual, k)
                for predicted, actual in zip(recommendations, targets)
            ])),
            "NDCG@K": float(np.mean([
                ndcg_at_k(predicted, actual, k)
                for predicted, actual in zip(recommendations, targets)
            ])),
            "MAP@K": map_at_k(recommendations, targets, k),
        })
    return pd.DataFrame(results).set_index("K")


def compare_baselines(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Vocabulary,
    targets: list[list[int]],
    model_metrics: pd.DataFrame,
    ks: tuple[int, ...] = (5, 10, 20),
    model_name: str = "GPT-2",
) -> pd.DataFrame:
    """Сравнить модель с общей популярностью и личными частотами.

    Args:
        train: Истории для расчёта общей популярности.
        val: Истории, добавляемые в личные частоты.
        test: Задаёт порядок пользователей; пустые истории пропускаются.
        vocab: Общий словарь кандидатов для всех стратегий.
        targets: Первые test-объекты пользователей с непустыми историями.
        model_metrics: Результат evaluate_metrics для модели на тех же примерах.
        ks: Значения K, совпадающие с model_metrics.
        model_name: Название модели в таблице сравнения.

    Returns:
        Таблица метрик с индексами Model и K.
    """
    max_k = max(ks)
    global_counts = Counter(
        int(item)
        for history in train.item_id
        for item in history
        if int(item) in vocab
    )
    popular_items = [item for item, count in global_counts.most_common(max_k)]
    popular_recommendations = [popular_items.copy() for _ in targets]

    histories = {row.uid: list(row.item_id) for row in train.itertuples(index=False)}
    for row in val.itertuples(index=False):
        histories[row.uid].extend(row.item_id)

    personal_recommendations = []
    for row in test.itertuples(index=False):
        if len(row.item_id) == 0:
            continue
        counts = Counter(int(item) for item in histories[row.uid] if int(item) in vocab)
        ranked = [item for item, count in counts.most_common(max_k)]
        # Дополняем личный топ общепопулярными треками без повторов.
        seen = set(ranked)
        ranked.extend(item for item in popular_items if item not in seen)
        personal_recommendations.append(ranked[:max_k])

    return pd.concat({
        "Popularity": evaluate_metrics(popular_recommendations, targets, ks),
        "Personal frequency": evaluate_metrics(personal_recommendations, targets, ks),
        model_name: model_metrics,
    }, names=["Model"])


def prepare_evaluation_contexts(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
) -> tuple[list[dict], list[list[int]]]:
    """Собрать реальные истории до первого test-события для ранжирования и награды.

    Args:
        train: Истории train с item_id, timestamp и played_ratio_pct.
        val: Истории validation, добавляемые в контекст.
        test: Истории test; пустые истории пропускаются.

    Returns:
        Контексты с полной историей и временем запроса, первые test-треки
        в исходных item_id. Отклики из test в контекст не попадают.
    """
    columns = ("item_id", "timestamp", "played_ratio_pct")
    histories = {
        row.uid: {name: np.asarray(getattr(row, name)) for name in columns}
        for row in train.itertuples(index=False)
    }
    for row in val.itertuples(index=False):
        for name in columns:
            histories[row.uid][name] = np.concatenate((histories[row.uid][name], getattr(row, name)))
    contexts, targets = [], []
    for row in test.itertuples(index=False):
        if not len(row.item_id):
            continue
        contexts.append({"history": histories[row.uid], "query_time": int(row.timestamp[0])})
        targets.append([int(row.item_id[0])])
    return contexts, targets


def evaluate_policy(
    model: GPT2LMHeadModel,
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    contexts: list[dict],
    targets: list[list[int]],
    statistics: dict,
    params: dict,
    ks: tuple[int, ...] = (5, 10, 20),
    batch_size: int = 32,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Оценить релевантность, жадный top-1 и поведение статического top-K.

    Args:
        model: Исходная либо RL-дообученная GPT-2.
        reference: Замороженная исходная GPT-2 для KL и reward model.
        head: Зафиксированный CatBoost.
        vocab: Общий словарь кандидатов.
        contexts: Полные реальные истории из prepare_evaluation_contexts.
        targets: Первые test-треки, включая неизвестные цели.
        statistics: Статистики только train.
        params: Одинаковая функция награды и сила штрафа для всех моделей.
        ks: Значения K для метрик релевантности.
        batch_size: Размер батча оценки.

    Returns:
        Recall/NDCG/MAP и словарь награды, KL и показателей поведения.
        Награда и ratio считаются для top-1, поведение — для top-max(ks).
        KL усредняется по тем же контекстам, без PAD/UNK.
    """
    model.eval()
    reference.eval()
    tokens = [vocab.encode(list(c["history"]["item_id"][-model.config.n_positions:])) for c in contexts]
    recommendations = recommend_top_k(model, tokens, vocab, max(ks), batch_size)
    metrics = evaluate_metrics(recommendations, targets, ks)
    rewards, ratios, divergences = [], [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(contexts), batch_size), desc="Reward and KL"):
            batch = contexts[start:start + batch_size]
            top_items = [row[0] for row in recommendations[start:start + batch_size]]
            actions = torch.tensor(vocab.encode(top_items), dtype=torch.long) - 2
            batch_rewards, batch_ratios = score_actions(reference, head, vocab, batch, actions, statistics, params)
            log_probs = policy_log_probs(model, batch, vocab)
            ref_probs = log_probs if model is reference else policy_log_probs(reference, batch, vocab)
            divergences.extend((log_probs.exp() * (log_probs - ref_probs)).sum(-1).cpu().tolist())
            rewards.extend(batch_rewards)
            ratios.extend(batch_ratios)
    counts = Counter(item for row in recommendations for item in row)
    n, k = len(recommendations), len(recommendations[0])
    histories = [set(c["history"]["item_id"]) for c in contexts]
    overlap = sum(count * (count - 1) for count in counts.values()) / (n * (n - 1) * k) if n > 1 else 0.0
    behavior = {
        "Predicted ratio@1": float(np.mean(ratios)), "Reward@1": float(np.mean(rewards)),
        "KL": float(np.mean(divergences)),
        f"Seen@{k} (%)": 100 * np.mean([item in history for row, history in zip(recommendations, histories) for item in row]),
        f"Catalog coverage@{k} (%)": 100 * len(counts) / (len(vocab) - 2),
        f"Overlap@{k} (%)": 100 * overlap,
        f"Mean popularity@{k}": np.mean([statistics["popularity"][item] for row in recommendations for item in row]),
    }
    return metrics, behavior


def recall_at_k(recommendations: list[int], targets: list[int], k: int = 10) -> float:
    """Вернуть долю уникальных целевых объектов, найденных в первых K рекомендациях."""
    targets = set(targets)
    if not targets or k <= 0:
        return 0.0
    return len(set(recommendations[:k]) & targets) / len(targets)


def ndcg_at_k(recommendations: list[int], targets: list[int], k: int = 10) -> float:
    """Вернуть NDCG@K с бинарной релевантностью для одного примера."""
    targets = set(targets)
    if not targets or k <= 0:
        return 0.0
    dcg = sum(
        1 / np.log2(rank + 1)
        for rank, item in enumerate(recommendations[:k], start=1)
        if item in targets
    )
    ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, min(k, len(targets)) + 1))
    return float(dcg / ideal_dcg)


def average_precision_at_k(recommendations: list[int], targets: list[int], k: int = 10) -> float:
    """Вернуть AP@K для одного примера, нормируя на min(K, число целевых объектов)."""
    targets = set(targets)
    if not targets or k <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommendations[:k], start=1):
        if item in targets:
            hits += 1
            score += hits / rank
    return score / min(k, len(targets))


def map_at_k(recommendations: list[list[int]], targets: list[list[int]], k: int = 10) -> float:
    """Усреднить AP@K по примерам; списки рекомендаций и целей должны совпадать по длине."""
    scores = [
        average_precision_at_k(predicted, actual, k)
        for predicted, actual in zip(recommendations, targets)
    ]
    return float(np.mean(scores)) if scores else 0.0
