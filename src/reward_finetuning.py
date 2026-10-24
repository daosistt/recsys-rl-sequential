from src.config import AdamW, DataLoader, Dataset, GPT2Config, GPT2LMHeadModel, Path, Ridge, json, np, pd, pickle, torch, tqdm
from src.evaluation import load_model
from src.reward_model import clear_memory, make_reward_examples
from src.vocabulary import Vocabulary


class RewardDataset(Dataset):
    """Последовательности «история + трек» с числовым откликом.

    Args:
        sequences: Истории в token_id, заканчивающиеся оцениваемым треком.
        targets: Доли прослушивания этих треков.
        seq_len: Максимальная длина последовательности.
        pad_idx: Индекс PAD в сохранённом словаре.
    """

    def __init__(self, sequences: list[list[int]], targets: np.ndarray, seq_len: int, pad_idx: int) -> None:
        self.sequences = sequences
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.pad_idx = pad_idx

    def __len__(self) -> int:
        """Вернуть число взаимодействий."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Вернуть историю с padding слева и целевой отклик."""
        sequence = self.sequences[idx]
        ids = torch.full((self.seq_len,), self.pad_idx, dtype=torch.long)
        ids[-len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": (ids != self.pad_idx).long(),
            "targets": self.targets[idx],
        }


class UnfrozenRewardModel(torch.nn.Module):
    """GPT-2 с обучаемой линейной головой, инициализированной из Ridge.

    Args:
        gpt: Копия сохранённого GPT-2 бейзлайна.
        ridge: Обученная Ridge или None при восстановлении всех весов из чекпоинта.
    """

    def __init__(self, gpt: GPT2LMHeadModel, ridge: Ridge | None = None) -> None:
        super().__init__()
        self.config = gpt.config
        self.backbone = gpt.transformer
        self.head = torch.nn.Linear(gpt.config.n_embd, 1)
        if ridge is not None:
            with torch.no_grad():
                self.head.weight.copy_(torch.as_tensor(ridge.coef_).reshape(1, -1))
                self.head.bias.fill_(float(ridge.intercept_))
        self.requires_grad_(True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Предсказать долю прослушивания последнего трека без ограничения диапазона."""
        positions = (attention_mask.cumsum(1) - 1).clamp(min=0)
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=positions, use_cache=False,
        ).last_hidden_state
        return self.head(hidden[:, -1]).squeeze(-1)


def load_unfrozen_reward(
    artifact_dir: Path, device: torch.device,
) -> tuple[UnfrozenRewardModel, dict]:
    """Загрузить лучший чекпоинт дообученной reward model без повторного обучения.

    Args:
        artifact_dir: Папка reward_unfrozen_minfreq5 с best.pt.
        device: Устройство для извлечения эмбеддингов или предсказания отклика.

    Returns:
        Замороженная модель и конфиги исходного эксперимента.
    """
    checkpoint = torch.load(artifact_dir / "best.pt", map_location="cpu", weights_only=True)
    # Состояние optimizer для извлечения признаков не нужно.
    checkpoint.pop("optimizer_state_dict", None)
    model = UnfrozenRewardModel(GPT2LMHeadModel(GPT2Config(**checkpoint["model_config"])))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    print(f"Reward checkpoint: эпоха {checkpoint['epoch']}, "
          f"validation MAE={checkpoint['best_val_mae']:.3f} п.п.")
    return model.to(device), checkpoint["experiment"]


def initialize_unfrozen_reward(
    baseline_dir: Path, ridge_dir: Path, device: torch.device,
) -> tuple[UnfrozenRewardModel, Vocabulary, dict, dict]:
    """Перенести веса GPT-2 и сохранённой Ridge в обучаемую нейросеть.

    Args:
        baseline_dir: Папка зафиксированного GPT-2 бейзлайна.
        ridge_dir: Папка Ridge, обученной на признаках этого бейзлайна.
        device: Устройство для дообучения.

    Returns:
        Модель, словарь и сохранённые конфиги GPT-2 и Ridge.
    """
    gpt, vocab, baseline_config = load_model(baseline_dir, torch.device("cpu"))
    with open(ridge_dir / "reward_head.pkl", "rb") as f:
        ridge = pickle.load(f)
    with open(ridge_dir / "config.json") as f:
        ridge_config = json.load(f)
    model = UnfrozenRewardModel(gpt, ridge).to(device)
    return model, vocab, baseline_config, ridge_config


def prepare_reward_loaders(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    vocab: Vocabulary,
    seq_len: int,
    sample_params: dict,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Подготовить те же случайные примеры, что использовались для Ridge/CatBoost.

    Args:
        train: Обучающие истории.
        val: Истории validation.
        test: Истории test.
        vocab: Сохранённый словарь бейзлайна.
        seq_len: Длина последовательности вместе с оцениваемым треком.
        sample_params: Размеры подвыборок и seed из эксперимента Ridge.
        batch_size: Размер батча для дообучения.

    Returns:
        Загрузчики train, val и test; перемешивается только train.
    """
    test_context = train[["uid", "item_id"]].copy()
    test_context["item_id"] = [list(a) + list(b) for a, b in zip(train.item_id, val.item_id)]
    generator = torch.Generator().manual_seed(sample_params["seed"])
    loaders = {}
    for name, data, context in (("train", train, None), ("val", val, train), ("test", test, test_context)):
        sequences, targets, _ = make_reward_examples(
            data, context, vocab, seq_len, sample_params[f"{name}_samples"], sample_params["seed"],
        )
        dataset = RewardDataset(sequences, targets, seq_len, vocab.pad_idx)
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train",
            generator=generator if name == "train" else None,
        )
    clear_memory()
    return loaders


def evaluate_unfrozen_reward(model: UnfrozenRewardModel, loader: DataLoader) -> dict[str, float]:
    """Посчитать MAE и RMSE модели отклика в процентных пунктах.

    Args:
        model: GPT-2 с регрессионной головой.
        loader: Загрузчик оцениваемых взаимодействий.

    Returns:
        Словарь MAE (п.п.) и RMSE (п.п.).
    """
    model.eval()
    device = next(model.parameters()).device
    absolute_error = squared_error = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            targets = batch.pop("targets")
            error = model(**batch) - targets
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            count += len(targets)
    return {"MAE (п.п.)": absolute_error / count * 100,
            "RMSE (п.п.)": (squared_error / count) ** 0.5 * 100}


def train_unfrozen_reward(
    model: UnfrozenRewardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: dict,
    output_dir: Path,
    experiment: dict,
) -> tuple[UnfrozenRewardModel, pd.DataFrame]:
    """Дообучить GPT-2 и голову по MSE, выбирая лучший чекпоинт по validation MAE.

    Args:
        model: Модель на выбранном устройстве, до создания optimizer.
        train_loader: Обучающие примеры.
        val_loader: Примеры validation; test при обучении не используется.
        params: Эпохи, learning rate для GPT-2/головы, weight decay и clipping.
        output_dir: Отдельная папка для чекпоинтов и истории обучения.
        experiment: Конфиги исходного бейзлайна, Ridge и подвыборок.

    Returns:
        Модель с лучшими весами и таблица истории, включая исходную validation-оценку.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": params["backbone_lr"]},
        {"params": model.head.parameters(), "lr": params["head_lr"]},
    ], weight_decay=params["weight_decay"])
    initial = evaluate_unfrozen_reward(model, val_loader)
    best_mae = initial["MAE (п.п.)"]
    history = [{"epoch": 0, "train_MSE": None,
                "val_MAE": best_mae, "val_RMSE": initial["RMSE (п.п.)"]}]

    def save_checkpoint(filename: str, epoch: int) -> None:
        """Сохранить веса, optimizer и настройки текущего эксперимента."""
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.to_dict(), "params": params,
            "experiment": experiment, "best_val_mae": best_mae, "history": history,
        }, output_dir / filename)

    # Если дообучение ухудшит качество, best.pt останется исходной Ridge-моделью.
    save_checkpoint("best.pt", 0)
    print(f"До разморозки: validation MAE={best_mae:.3f} п.п.")
    for epoch in tqdm(range(1, params["num_epochs"] + 1), desc="Reward epochs"):
        model.train()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            targets = batch.pop("targets")
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(**batch), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
            optimizer.step()
            total_loss += loss.item() * len(targets)
            count += len(targets)

        validation = evaluate_unfrozen_reward(model, val_loader)
        history.append({"epoch": epoch, "train_MSE": total_loss / count,
                        "val_MAE": validation["MAE (п.п.)"], "val_RMSE": validation["RMSE (п.п.)"]})
        if validation["MAE (п.п.)"] < best_mae:
            best_mae = validation["MAE (п.п.)"]
            save_checkpoint("best.pt", epoch)
        save_checkpoint("latest.pt", epoch)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"Epoch {epoch}: train MSE={total_loss / count:.4f}, "
              f"val MAE={validation['MAE (п.п.)']:.3f} п.п.")
        clear_memory()

    # Освобождаем optimizer до чтения чекпоинта, чтобы снизить пиковый расход памяти.
    del optimizer
    clear_memory()
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Загружена лучшая эпоха: {checkpoint['epoch']}")
    model.eval()
    return model, pd.DataFrame(history)
