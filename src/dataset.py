from src.config import Dataset, pd, torch
from src.vocabulary import Vocabulary


class RecDataset(Dataset):
    """Датасет пользовательских историй для GPT-2.

    Соседние окна пересекаются на один токен, чтобы сохранить переход
    между окнами. Истории короче двух токенов пропускаются.

    Args:
        data: Таблица пользовательских историй с колонкой item_id.
        vocab: Словарь, построенный по train.
        seq_len: Длина окна, включая padding; не меньше двух.
        context: Предыдущие истории пользователей, например train для validation.
            Они используются как контекст и исключаются из loss.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vocab: Vocabulary,
        seq_len: int = 128,
        context: pd.DataFrame | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len должна быть не меньше 2")

        self.seq_len = seq_len
        self.pad_idx = vocab.pad_idx
        context_histories = {} if context is None else context.set_index("uid").item_id.to_dict()
        self.token_ids = []
        self.context_lengths = []
        for row in data.itertuples(index=False):
            if len(row.item_id) == 0:
                continue
            prefix = [] if context is None else list(context_histories.get(row.uid, []))[-(seq_len - 1):]
            history = prefix + list(row.item_id)
            if len(history) < 2:
                continue
            self.token_ids.append(torch.tensor(vocab.encode(history), dtype=torch.long))
            self.context_lengths.append(len(prefix))
        self.windows = [
            (user_idx, start)
            for user_idx, history in enumerate(self.token_ids)
            for start in range(0, len(history) - 1, seq_len - 1)
        ]

    def __len__(self) -> int:
        """Вернуть количество окон."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Вернуть окно, маску внимания и цели с padding, исключённым из loss."""
        user_idx, start = self.windows[idx]
        chunk = self.token_ids[user_idx][start:start + self.seq_len]

        input_ids = torch.full((self.seq_len,), self.pad_idx, dtype=torch.long)
        input_ids[:len(chunk)] = chunk
        attention_mask = (input_ids != self.pad_idx).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:max(0, self.context_lengths[user_idx] - start)] = -100
        labels[0] = -100  # Первая позиция окна не имеет предыдущего токена.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
