from src.config import (
    plt,
    clear_output,
    GPT2LMHeadModel,
    DataLoader,
    torch,
    device,
    ARTIFACTS_DIR,
    Path
)

def plot_loss(train_loss: list[float], val_loss: list[float]) -> None:
    """Показать train и validation loss каждого батча."""

    clear_output(True)
    plt.figure(figsize=(18, 6))

    # Лосс на обучающей выборке
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train")
    plt.title("Train Loss")
    plt.xlabel("Batch")
    plt.ylabel("loss")
    plt.grid(True)

    # Лосс на валидационной выборке
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Valid")
    plt.legend()
    plt.title("Validation Loss")
    plt.xlabel("Batch")
    plt.ylabel("loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close()

def train_loop(
    model: GPT2LMHeadModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device = device,
    num_epochs: int = 5,
    checkpoint_dir: Path = ARTIFACTS_DIR / "checkpoints",
    plot_every: int = 50,
) -> dict[str, list[float]]:
    """Обучить модель с валидацией после каждой эпохи.

    Args:
        model: GPT2LMHeadModel.
        train_loader: DataLoader с input_ids, attention_mask и labels.
        val_loader: DataLoader с контекстом из train и целями из validation.
        optimizer: Оптимизатор модели.
        device: Устройство для вычислений.
        num_epochs: Количество эпох.
        checkpoint_dir: Папка для latest.pt и лучшего по validation loss best.pt.
        plot_every: Обновлять графики каждые N батчей в train и validation.

    Returns:
        Loss каждого батча (train, val) и средние по целевым токенам
        значения за эпоху (train_epoch, val_epoch).
    """
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    loss_history = {"train": [], "val": [], "train_epoch": [], "val_epoch": []}

    for epoch in range(num_epochs):

        # Train
        model.train()
        train_loss = 0.0
        train_tokens = 0
        for batch in train_loader:

            batch = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()  

            num_tokens = (batch["labels"][:, 1:] != -100).sum().item()
            batch_loss = loss.item()
            loss_history["train"].append(batch_loss)
            train_loss += batch_loss * num_tokens
            train_tokens += num_tokens

            if len(loss_history["train"]) % plot_every == 0:
                plot_loss(loss_history["train"], loss_history["val"])

        mean_loss = train_loss / train_tokens
        loss_history["train_epoch"].append(mean_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:

                batch = {
                    name: tensor.to(device)
                    for name, tensor in batch.items()
                }

                outputs = model(**batch)
                num_tokens = (batch["labels"][:, 1:] != -100).sum().item()
                batch_loss = outputs.loss.item()
                loss_history["val"].append(batch_loss)
                val_loss += batch_loss * num_tokens
                val_tokens += num_tokens

                if len(loss_history["val"]) % plot_every == 0:
                    plot_loss(loss_history["train"], loss_history["val"])

        mean_val_loss = val_loss / val_tokens
        loss_history["val_epoch"].append(mean_val_loss)

        is_best = mean_val_loss < best_val_loss
        best_val_loss = min(best_val_loss, mean_val_loss)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "loss_history": loss_history,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        plot_loss(loss_history["train"], loss_history["val"])
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train loss = {mean_loss:.4f}, val loss = {mean_val_loss:.4f}"
        )

    return loss_history
