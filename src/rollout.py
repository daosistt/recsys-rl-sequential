from src.config import CatBoostRegressor, Counter, GPT2LMHeadModel, np, pd, torch, tqdm
from src.reinforce import policy_log_probs
from src.reward import (
    make_candidate_features, popularity_penalty_reward, predict_listening_ratio,
)
from src.vocabulary import Vocabulary


def generate_rollouts(
    model: GPT2LMHeadModel,
    contexts: list[dict],
    vocab: Vocabulary,
    num_steps: int = 5,
    batch_size: int = 32,
    avoid_repeats: bool = False,
) -> list[list[int]]:
    """Последовательно выбрать треки жадно, добавляя каждый в контекст модели.

    Args:
        model: Исходная или RL-дообученная GPT-2.
        contexts: Исходные реальные истории и времена запросов.
        vocab: Общий словарь моделей.
        num_steps: Длина сгенерированного списка.
        batch_size: Число пользователей в батче.
        avoid_repeats: Исключать уже выбранные треки внутри генерируемого списка.
            Треки из реальной истории пользователя остаются допустимыми.

    Returns:
        Списки исходных item_id в порядке генерации. PAD и UNK исключены,
        повторы разрешены по умолчанию. Исходные контексты и реальные отклики
        не изменяются. При avoid_repeats длина не должна превышать размер каталога.
    """
    if avoid_repeats and num_steps > len(vocab) - 2:
        raise ValueError("Для списка без повторов нужно не меньше num_steps треков в словаре")
    model.eval()
    sequences = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(contexts), batch_size), desc="Generate rollouts"):
            batch = contexts[start:start + batch_size]
            current = [
                {"history": {"item_id": list(c["history"]["item_id"][-model.config.n_positions:])}}
                for c in batch
            ]
            generated = [[] for _ in batch]
            for step in range(num_steps):
                log_probs = policy_log_probs(model, current, vocab)
                if avoid_repeats and step > 0:
                    previous = torch.tensor(
                        [vocab.encode(sequence) for sequence in generated],
                        dtype=torch.long, device=log_probs.device,
                    ) - 2
                    log_probs.scatter_(1, previous, -float("inf"))
                actions = log_probs.argmax(dim=-1)
                items = vocab.decode((actions.cpu() + 2).tolist())
                for context, sequence, item in zip(current, generated, items):
                    sequence.append(item)
                    context["history"]["item_id"].append(item)
            sequences.extend(generated)
    return sequences


def evaluate_rollouts(
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    contexts: list[dict],
    sequences: list[list[int]],
    statistics: dict,
    targets: list[list[int]] | None = None,
    penalty: float = 0.1,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Оценить генерацию по шагам без симуляции неизвестных откликов пользователя.

    Args:
        reference: Замороженная исходная GPT-2 внутри reward model.
        head: Зафиксированный CatBoost на эмбеддингах и пяти признаках.
        vocab: Общий словарь моделей.
        contexts: Реальные истории до начала test с фиксированным временем запроса.
        sequences: Результат generate_rollouts; одинаковая длина для всех пользователей.
        statistics: Статистики train.
        targets: Первые фактические test-треки для оценки попадания в префикс,
            включая неизвестные цели; не будущие отклики на сгенерированные треки.
        penalty: Общая сила штрафов при сравнении всех моделей.
        batch_size: Размер батча расчёта награды.

    Returns:
        Таблица по шагам. Пять признаков и время запроса остаются основанными
        на реальной истории. Эмбеддинги учитывают ранее сгенерированные треки;
        контрольный прогноз использует только реальный контекст для тех же
        кандидатов. Их разница показывает чувствительность к контексту,
        а не измеренную ошибку reward model или реальный отклик пользователя.
    """
    reference.requires_grad_(False)
    real_histories = [list(c["history"]["item_id"][-reference.config.n_positions:]) for c in contexts]
    real_sets = [set(c["history"]["item_id"]) for c in contexts]
    rows = []
    for step in tqdm(range(len(sequences[0])), desc="Evaluate rollout steps"):
        items = [sequence[step] for sequence in sequences]
        generated_prefixes = [sequence[:step] for sequence in sequences]
        current_histories = [history + prefix for history, prefix in zip(real_histories, generated_prefixes)]
        ratios, real_context_ratios = [], []
        for start in range(0, len(contexts), batch_size):
            stop = start + batch_size
            batch = contexts[start:stop]
            tabular = make_candidate_features(
                [c["history"] for c in batch], items[start:stop],
                [c["query_time"] for c in batch], statistics,
            )
            predicted = predict_listening_ratio(
                reference, head, vocab, current_histories[start:stop],
                items[start:stop], tabular, batch_size,
            )
            control = predicted if step == 0 else predict_listening_ratio(
                reference, head, vocab, real_histories[start:stop],
                items[start:stop], tabular, batch_size,
            )
            ratios.extend(predicted)
            real_context_ratios.extend(control)
        ratios, real_context_ratios = np.asarray(ratios), np.asarray(real_context_ratios)
        counts = Counter(items)
        repeat_flags = np.asarray([
            item in history or item in prefix
            for item, history, prefix in zip(items, real_sets, generated_prefixes)
        ])
        row = {
            "Step": step + 1,
            "Predicted ratio": float(ratios.mean()),
            "Real-context ratio": float(real_context_ratios.mean()),
            "Context shift (absolute)": float(np.abs(ratios - real_context_ratios).mean()),
            "Repeat-penalized reward": float((ratios - penalty * repeat_flags).mean()),
            "Popularity-penalized reward": float(popularity_penalty_reward(ratios, items, statistics, penalty).mean()),
            "Seen in real history (%)": 100 * np.mean([item in history for item, history in zip(items, real_sets)]),
            "Repeated in generation (%)": 100 * np.mean([item in prefix for item, prefix in zip(items, generated_prefixes)]),
            "Mean popularity": np.mean([statistics["popularity"][item] for item in items]),
            "Mean duration (s)": np.mean([statistics["lengths"][item] for item in items]),
            "Catalog coverage (%)": 100 * len(counts) / (len(vocab) - 2),
            "Most common track (%)": 100 * max(counts.values()) / len(items),
        }
        if targets is not None:
            row["First-target hit@prefix (%)"] = 100 * np.mean([
                bool(set(sequence[:step + 1]) & set(actual))
                for sequence, actual in zip(sequences, targets)
            ])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Step")
