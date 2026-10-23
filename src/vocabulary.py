from src.config import Counter, pd


class Vocabulary:
    """Словарь для преобразования item_id в индексы токенов и обратно.

    Args:
        train: Таблица обучающих историй с колонкой item_id.
        min_frequency: Минимальное число появлений объекта в train.
    """

    def __init__(self, train: pd.DataFrame, min_frequency: int = 1) -> None:
        self.token_to_item: list[int | str] = []
        self.item_to_token: dict[int | str, int] = {}
        self._SPECIAL_TOKENS = ["<pad>", "<unk>"]
        item_freq = Counter(
            int(item_id)
            for history in train.item_id
            for item_id in history
        )
        self._build_vocab(item_freq, min_frequency)

    def _build_vocab(self, item_freq: dict[int, int], min_frequency: int) -> None:
        for token in self._SPECIAL_TOKENS:
            self._append_token(token)

        for item_id, frequency in item_freq.items():
            if frequency >= min_frequency:
                self._append_token(int(item_id))

    def _append_token(self, token: int | str) -> None:
        if token not in self.item_to_token:
            self.item_to_token[token] = len(self.token_to_item)
            self.token_to_item.append(token)

    def __getitem__(self, token: int | str) -> int:
        """Вернуть индекс токена или UNK для неизвестного объекта."""
        return self.item_to_token.get(token, self.unk_idx)

    def __len__(self) -> int:
        """Вернуть размер словаря, включая специальные токены."""
        return len(self.token_to_item)

    def __contains__(self, token: int | str) -> bool:
        """Проверить наличие токена в словаре."""
        return token in self.item_to_token

    @property
    def unk_idx(self) -> int:
        """Индекс токена UNK."""
        return self.item_to_token["<unk>"]

    @property
    def pad_idx(self) -> int:
        """Индекс токена PAD."""
        return self.item_to_token["<pad>"]

    def encode(self, item_ids: list[int]) -> list[int]:
        """Преобразовать item_id в токены, заменяя неизвестные объекты на UNK."""
        return [self[item_id] for item_id in item_ids]

    def decode(self, indices: list[int], skip_special: bool = False) -> list[int | str]:
        """Восстановить item_id, исключая спецтокены при skip_special=True."""
        tokens = [self.token_to_item[idx] if 0 <= idx < len(self.token_to_item) else "<unk>" for idx in indices]
        if skip_special:
            tokens = [token for token in tokens if token not in self._SPECIAL_TOKENS]
        return tokens
