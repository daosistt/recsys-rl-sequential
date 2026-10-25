from src.config import AdamW, CatBoostRegressor, DataLoader, GPT2LMHeadModel, np, torch, tqdm
from src.reinforce import policy_log_probs, score_actions
from src.vocabulary import Vocabulary


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """Вернуть суммы текущей и будущих наград для массива [batch, steps], gamma=1."""
    return np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1).copy()


def rollout_log_probs(
    model: GPT2LMHeadModel,
    contexts: list[dict],
    generated: list[list[int]],
    vocab: Vocabulary,
    avoid_repeats: bool,
) -> torch.Tensor:
    """Посчитать распределение следующего трека с учётом сгенерированного префикса.

    Args:
        model: Обучаемая политика или исходный бейзлайн.
        contexts: Реальные истории до начала генерации.
        generated: Ранее выбранные треки для каждого пользователя.
        vocab: Общий словарь.
        avoid_repeats: Исключать треки префикса, сохраняя знакомые треки истории.

    Returns:
        Логарифмы вероятностей реальных треков. После маскирования распределение
        заново нормируется. На исключённых позициях остаётся минус бесконечность.
    """
    current = [
        {"history": {"item_id": list(c["history"]["item_id"][-model.config.n_positions:]) + prefix}}
        for c, prefix in zip(contexts, generated)
    ]
    log_probs = policy_log_probs(model, current, vocab)
    if avoid_repeats and generated[0]:
        previous = torch.tensor(
            [vocab.encode(prefix) for prefix in generated], dtype=torch.long, device=log_probs.device,
        ) - 2
        log_probs = log_probs.scatter(1, previous, -float("inf")).log_softmax(dim=-1)
    return log_probs


def rollout_kl(log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
    """Посчитать KL по разрешённым трекам, исключая выражение 0 * (-inf - -inf)."""
    excluded = torch.isneginf(log_probs)
    difference = log_probs.masked_fill(excluded, 0) - reference_log_probs.masked_fill(excluded, 0)
    return (log_probs.exp() * difference).sum(dim=-1)


def sample_trajectory(
    model: GPT2LMHeadModel,
    reference: GPT2LMHeadModel,
    head: CatBoostRegressor,
    vocab: Vocabulary,
    contexts: list[dict],
    statistics: dict,
    params: dict,
    generator: torch.Generator,
    greedy: bool = False,
) -> dict:
    """Сгенерировать и оценить траектории без хранения графов пяти шагов.

    Args:
        model: Политика для сэмплирования либо reference для жадного baseline.
        reference: Замороженная GPT-2 для KL и reward model.
        head: Замороженный CatBoost.
        vocab: Общий словарь.
        contexts: Исходные реальные истории.
        statistics: Статистики только train.
        params: num_steps, avoid_repeats и настройки награды.
        generator: Генератор случайных действий.
        greedy: Выбирать argmax вместо сэмплирования для контекстного baseline.

    Returns:
        Последовательности item_id, индексы действий, награды, ratio, KL и
        выбранные log-probabilities. Числовые массивы имеют форму [batch, steps].
        Реальные отклики, статистики и время запроса не изменяются.
    """
    model.eval()
    reference.eval()
    generated = [[] for _ in contexts]
    values = {key: [] for key in ("actions", "rewards", "ratios", "kl", "log_probs")}
    with torch.no_grad():
        for _ in range(params["num_steps"]):
            log_probs = rollout_log_probs(model, contexts, generated, vocab, params["avoid_repeats"])
            reference_log_probs = log_probs if model is reference else rollout_log_probs(
                reference, contexts, generated, vocab, params["avoid_repeats"],
            )
            actions = log_probs.argmax(-1) if greedy else torch.multinomial(
                log_probs.exp(), 1, generator=generator,
            ).squeeze(1)
            rewards, ratios = score_actions(
                reference, head, vocab, contexts, actions, statistics, params, generated=generated,
            )
            values["actions"].append(actions.cpu().numpy())
            values["rewards"].append(rewards)
            values["ratios"].append(ratios)
            values["kl"].append(rollout_kl(log_probs, reference_log_probs).cpu().numpy())
            values["log_probs"].append(log_probs.gather(1, actions[:, None]).squeeze(1).cpu().numpy())
            for prefix, item in zip(generated, vocab.decode((actions.cpu() + 2).tolist())):
                prefix.append(item)
    return {"sequences": generated, **{key: np.stack(value, axis=1) for key, value in values.items()}}


def run_multistep_epoch(
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
    """Обучить на накопленной пятишаговой награде либо оценить validation.

    Args:
        policy: Обучаемая GPT-2; dropout отключён, градиенты разрешены.
        reference: Отдельная замороженная исходная GPT-2.
        head: Неизменяемый CatBoost.
        vocab: Общий словарь.
        contexts: Train-контексты при обучении или фиксированные validation-контексты.
        statistics: Статистики train.
        params: Настройки многошагового RL, включая num_steps и avoid_repeats.
        seed: Seed батчей и действий.
        optimizer: AdamW либо None для оценки без обновления весов.

    Returns:
        Средние reward, ratio, reference_reward и KL на шаг, total_reward за
        весь список, loss и objective = reward - kl_coef * KL. Будущие награды
        и KL входят в return-to-go. Градиент точного KL дополнительно считается
        на каждом посещённом состоянии. Нормировка на num_steps сохраняет
        шкалу objective относительно одношагового эксперимента.
    """
    if params["avoid_repeats"] and params["num_steps"] > len(vocab) - 2:
        raise ValueError("Для списка без повторов недостаточно треков в словаре")
    policy.eval()
    reference.eval()
    reference.requires_grad_(False)
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(
        range(len(contexts)), batch_size=params["batch_size"], shuffle=optimizer is not None,
        generator=torch.Generator().manual_seed(seed),
    )
    totals = {key: 0.0 for key in ("reward", "ratio", "reference_reward", "total_reward", "kl", "loss", "objective")}
    count = 0
    for indices in tqdm(loader, desc="RL 5-step train" if optimizer is not None else "RL 5-step validation", leave=False):
        batch = [contexts[int(i)] for i in indices]
        trajectory = sample_trajectory(policy, reference, head, vocab, batch, statistics, params, generator)
        baseline = sample_trajectory(reference, reference, head, vocab, batch, statistics, params, generator, greedy=True)
        regularized = trajectory["rewards"] - params["kl_coef"] * trajectory["kl"]
        # Жадная reference-траектория зависит только от начальной истории,
        # а не от случайно выбранных действий обучаемой политики.
        advantages = (returns_to_go(regularized) - returns_to_go(baseline["rewards"])) / params["num_steps"]
        loss_value = float(-(advantages * trajectory["log_probs"]).sum(1).mean()
                           + params["kl_coef"] * trajectory["kl"].mean())
        if optimizer is not None:
            optimizer.zero_grad()
            # Пересчитываем и освобождаем граф каждого шага отдельно. Веса
            # остаются прежними до конца батча, поэтому действия ещё on-policy.
            for step in range(params["num_steps"]):
                prefixes = [sequence[:step] for sequence in trajectory["sequences"]]
                with torch.no_grad():
                    reference_log_probs = rollout_log_probs(reference, batch, prefixes, vocab, params["avoid_repeats"])
                log_probs = rollout_log_probs(policy, batch, prefixes, vocab, params["avoid_repeats"])
                actions = torch.as_tensor(trajectory["actions"][:, step], dtype=torch.long, device=device)
                advantage = torch.as_tensor(advantages[:, step], dtype=log_probs.dtype, device=device)
                selected = log_probs.gather(1, actions[:, None]).squeeze(1)
                kl = rollout_kl(log_probs, reference_log_probs).mean()
                loss = -(advantage * selected).mean() + params["kl_coef"] * kl / params["num_steps"]
                loss.backward()
                del loss, kl, selected, log_probs, reference_log_probs
            torch.nn.utils.clip_grad_norm_(policy.parameters(), params["max_grad_norm"])
            optimizer.step()
        values = {
            "reward": float(trajectory["rewards"].mean()), "ratio": float(trajectory["ratios"].mean()),
            "reference_reward": float(baseline["rewards"].mean()),
            "total_reward": float(trajectory["rewards"].sum(1).mean()),
            "kl": float(trajectory["kl"].mean()), "loss": loss_value,
            "objective": float(regularized.mean()),
        }
        for key, value in values.items():
            totals[key] += value * len(batch)
        count += len(batch)
    return {key: value / count for key, value in totals.items()}
