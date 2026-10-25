from src.config import AdamW, Callable, CatBoostRegressor, DataLoader, GPT2LMHeadModel, Path, np, pd, pickle, torch, tqdm, json
from src.reward import (
    listening_reward, make_candidate_features, popularity_penalty_reward,
    predict_listening_ratio, repeat_penalty_reward,
)
from src.reward_model import clear_memory, make_reward_examples
from src.vocabulary import Vocabulary


def prepare_rl_contexts(
    data: pd.DataFrame,
    vocab: Vocabulary,
    max_samples: int,
    seed: int = 42,
    context: pd.DataFrame | None = None,
) -> list[dict]:
    """Выбрать контексты перед реальными событиями, не используя текущий отклик.

    Args:
        data: Train или validation; test в RL-обучение не передаётся.
        vocab: Словарь исходной GPT-2.
        max_samples: Число случайных контекстов перед известными целевыми треками.
        seed: Seed подвыборки.
        context: Предыдущая часть истории: train для validation, None для train.

    Returns:
        Список историй с item_id, timestamp, played_ratio_pct и временем запроса.
        Массивы истории заканчиваются перед выбранным событием, без его отклика.
    """
    sequences, targets, samples = make_reward_examples(data, context, vocab, 2, max_samples, seed)
    del sequences, targets
    prefixes = {} if context is None else {row.uid: row for row in context.itertuples(index=False)}
    rows = list(data.itertuples(index=False))
    histories = {}
    for user in np.unique(samples[:, 0]):
        row = rows[user]
        prefix = prefixes.get(row.uid)
        histories[user] = {
            name: np.asarray(getattr(row, name)) if prefix is None else np.concatenate(
                (getattr(prefix, name), getattr(row, name)),
            )
            for name in ("item_id", "timestamp", "played_ratio_pct")
        }
    contexts = []
    for user, position in samples:
        row = rows[user]
        prefix = prefixes.get(row.uid)
        stop = position + (len(prefix.item_id) if prefix is not None else 0)
        contexts.append({
            "history": {name: values[:stop] for name, values in histories[user].items()},
            "query_time": int(row.timestamp[position]),
        })
    return contexts


def policy_log_probs(
    model: GPT2LMHeadModel, contexts: list[dict], vocab: Vocabulary,
) -> torch.Tensor:
    """Получить логарифмы вероятностей реальных треков с исключением PAD и UNK.

    Args:
        model: Обучаемая политика или замороженная исходная GPT-2.
        contexts: Истории из prepare_rl_contexts.
        vocab: Общий словарь с PAD=0, UNK=1.

    Returns:
        Тензор [batch, число реальных треков]; колонка j соответствует токену j+2.
        Градиенты сохраняются, если вызов сделан вне torch.no_grad.
    """
    device = next(model.parameters()).device
    sequences = [vocab.encode(list(c["history"]["item_id"][-model.config.n_positions:])) for c in contexts]
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), vocab.pad_idx, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        ids[i, -len(sequence):] = torch.tensor(sequence, device=device)
    mask = (ids != vocab.pad_idx).long()
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    logits = model(
        input_ids=ids, attention_mask=mask, position_ids=positions,
        logits_to_keep=1, use_cache=False,
    ).logits[:, -1, 2:]
    return logits.log_softmax(dim=-1)


def score_actions(
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    contexts: list[dict],
    actions: torch.Tensor,
    statistics: dict,
    params: dict,
    generated: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Оценить выбранные треки замороженной reward model и применить награду.

    Args:
        reference: Замороженная исходная GPT-2 для эмбеддингов CatBoost.
        head: Сохранённый CatBoost с пятью дополнительными признаками.
        vocab: Словарь исходной модели.
        contexts: Истории до выбора треков.
        actions: Индексы колонок policy_log_probs; token_id = action + 2.
        statistics: Статистики только train из prepare_track_statistics.
        params: reward_type, penalty и batch_size.
        generated: Ранее выбранные треки. Добавляются в контекст эмбеддингов
            и проверку повторов, но не в историю наблюдённых откликов.

    Returns:
        Итоговые награды и исходные прогнозы доли прослушивания, без градиентов.
    """
    items = vocab.decode((actions.detach().cpu() + 2).tolist())
    histories = [c["history"] for c in contexts]
    tabular = make_candidate_features(histories, items, [c["query_time"] for c in contexts], statistics)
    prefixes = generated if generated is not None else [[] for _ in contexts]
    item_histories = [list(history["item_id"]) + prefix for history, prefix in zip(histories, prefixes)]
    ratios = predict_listening_ratio(reference, head, vocab, item_histories, items, tabular, params["batch_size"])
    reward_type = params["reward_type"]
    if reward_type == "listening":
        rewards = listening_reward(ratios)
    elif reward_type == "repeat_penalty":
        rewards = repeat_penalty_reward(ratios, item_histories, items, params["penalty"])
    elif reward_type == "popularity_penalty":
        rewards = popularity_penalty_reward(ratios, items, statistics, params["penalty"])
    else:
        raise ValueError(f"Неизвестная функция награды: {reward_type}")
    return rewards, ratios


def run_reinforce_epoch(
    policy: GPT2LMHeadModel,
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    contexts: list[dict],
    statistics: dict,
    params: dict,
    seed: int,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    """Пройти контексты с REINFORCE либо оценить фиксированную validation-выборку.

    Args:
        policy: GPT-2, вероятности которой оптимизируем.
        reference: Отдельная неизменяемая исходная GPT-2 для KL и reward model.
        head: Неизменяемый CatBoost.
        vocab: Общий словарь моделей.
        contexts: Только train при обучении, validation при оценке.
        statistics: Статистики train для расчёта награды.
        params: Настройки награды, KL, батчей и clipping.
        seed: Seed перемешивания и сэмплирования треков.
        optimizer: AdamW для обновления политики; None означает оценку без градиентов.

    Returns:
        Средние reward, ratio, reference_reward, KL, loss и reward - kl_coef * KL.
        Награды в долях; KL — точный KL(policy || reference) по реальным трекам.
    """
    device = next(policy.parameters()).device
    # Отключаем dropout и при обучении: вероятности выбора и градиента совпадают.
    policy.eval()
    reference.eval()
    reference.requires_grad_(False)
    loader = DataLoader(
        range(len(contexts)), batch_size=params["batch_size"], shuffle=optimizer is not None,
        generator=torch.Generator().manual_seed(seed),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {name: 0.0 for name in ("reward", "ratio", "reference_reward", "kl", "loss", "objective")}
    count = 0
    for indices in tqdm(loader, desc="RL train" if optimizer is not None else "RL validation", leave=False):
        batch = [contexts[int(i)] for i in indices]
        with torch.no_grad():
            reference_log_probs = policy_log_probs(reference, batch, vocab)
            greedy_actions = reference_log_probs.argmax(dim=-1)
        with torch.set_grad_enabled(optimizer is not None):
            log_probs = policy_log_probs(policy, batch, vocab)
            actions = torch.multinomial(log_probs.detach().exp(), 1, generator=generator).squeeze(1)
            rewards, ratios = score_actions(reference, head, vocab, batch, actions, statistics, params)
            baseline_rewards, _ = score_actions(reference, head, vocab, batch, greedy_actions, statistics, params)
            # Контекстный baseline не зависит от случайного действия политики.
            advantage = torch.as_tensor(rewards - baseline_rewards, dtype=log_probs.dtype, device=device)
            selected_log_probs = log_probs.gather(1, actions[:, None]).squeeze(1)
            kl = (log_probs.exp() * (log_probs - reference_log_probs)).sum(dim=-1).mean()
            loss = -(advantage * selected_log_probs).mean() + params["kl_coef"] * kl
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), params["max_grad_norm"])
                optimizer.step()
        values = {
            "reward": float(rewards.mean()), "ratio": float(ratios.mean()),
            "reference_reward": float(baseline_rewards.mean()), "kl": kl.item(), "loss": loss.item(),
            "objective": float(rewards.mean()) - params["kl_coef"] * kl.item(),
        }
        for name, value in values.items():
            totals[name] += value * len(batch)
        count += len(batch)
    return {name: value / count for name, value in totals.items()}


def train_reinforce(
    policy: GPT2LMHeadModel,
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    train_contexts: list[dict],
    val_contexts: list[dict],
    statistics: dict,
    params: dict,
    output_dir: Path,
    experiment: dict,
    epoch_runner: Callable = run_reinforce_epoch,
) -> pd.DataFrame:
    """Дообучить GPT-2 и сохранить лучший по validation objective вариант.

    Args:
        policy: Свежая копия исходной GPT-2 для каждого варианта награды.
        reference: Отдельная замороженная копия того же бейзлайна.
        head: Сохранённый CatBoost; не обучается.
        vocab: Общий словарь.
        train_contexts: Контексты только из train.
        val_contexts: Фиксированные контексты validation, без test.
        statistics: Статистики train.
        params: RL_PARAMS с выбранной наградой и силой KL-штрафа.
        output_dir: Отдельная папка RL-запуска.
        experiment: Конфиг исходного бейзлайна и имена исходных артефактов.
        epoch_runner: Одношаговый или многошаговый проход по контекстам.

    Returns:
        История train/validation, включая epoch=0 до RL. Политика восстанавливается
        из лучшего чекпоинта. Validation objective — модельная оценка, не релевантность.
    """
    if policy is reference:
        raise ValueError("Для политики и reward model нужны отдельные копии GPT-2")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    saved_config = {**experiment["baseline_config"], "rl": params, "sources": experiment}
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    policy.requires_grad_(True)
    optimizer = AdamW(policy.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    validation = epoch_runner(
        policy, reference, head, vocab, val_contexts, statistics, params, params["seed"],
    )
    best_objective = validation["objective"]
    history = [{"epoch": 0, **{f"val_{name}": value for name, value in validation.items()}}]

    def save_checkpoint(name: str, epoch: int) -> None:
        """Сохранить веса и настройки через временный файл."""
        temporary = checkpoint_dir / f"{name}.tmp"
        torch.save({
            "epoch": epoch, "model_state_dict": policy.state_dict(),
            "model_config": policy.config.to_dict(), "params": params,
            "best_val_objective": best_objective, "history": history,
        }, temporary)
        temporary.replace(checkpoint_dir / name)

    save_checkpoint("best.pt", 0)
    for epoch in tqdm(range(1, params["num_epochs"] + 1), desc="RL epochs"):
        training = epoch_runner(
            policy, reference, head, vocab, train_contexts, statistics, params, params["seed"] + epoch, optimizer,
        )
        validation = epoch_runner(
            policy, reference, head, vocab, val_contexts, statistics, params, params["seed"],
        )
        history.append({"epoch": epoch,
                        **{f"train_{name}": value for name, value in training.items()},
                        **{f"val_{name}": value for name, value in validation.items()}})
        if validation["objective"] > best_objective:
            best_objective = validation["objective"]
            save_checkpoint("best.pt", epoch)
        save_checkpoint("latest.pt", epoch)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"Epoch {epoch}: val reward={validation['reward']:.4f}, "
              f"KL={validation['kl']:.4f}, objective={validation['objective']:.4f}")
    del optimizer
    clear_memory()
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return pd.DataFrame(history)
