from src.config import GPT2Config, GPT2LMHeadModel, Path, json, np, pd, pickle, torch, tqdm
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
