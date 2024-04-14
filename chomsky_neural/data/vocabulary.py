from collections import defaultdict
from typing import Dict, Optional, Union, overload

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
LABEL_PAD_INDEX = -1


class Vocabulary:
    def __init__(
        self,
        token2index: Optional[Dict[str, Dict[str, int]]] = None,
        add_unk_token: bool = True,
        add_pad_token: bool = True,
    ) -> None:
        self._token2index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._index2token: Dict[str, Dict[int, str]] = defaultdict(dict)
        if token2index is not None:
            for namespace in token2index:
                self._token2index[namespace] = {
                    key: value for key, value in token2index[namespace].items()
                }
                self._index2token[namespace] = {
                    value: key for key, value in token2index[namespace].items()
                }

    def add(self, token: str, namespace: str = "tokens") -> int:
        if token not in self._token2index[namespace]:
            index = len(self._token2index[namespace])
            self._token2index[namespace][token] = index
            self._index2token[namespace][index] = token
        return self.get(token, namespace)

    @overload
    def get(self, token: str, namespace: str = "tokens") -> int:
        ...

    @overload
    def get(self, index: int, namespace: str = "tokens") -> str:
        ...

    def get(
        self, token_or_index: Union[str, int], namespace: str = "tokens"
    ) -> Union[str, int]:
        if isinstance(token_or_index, str):
            if token_or_index in self._token2index[namespace]:
                return self._token2index[namespace][token_or_index]
            else:
                return self._token2index[namespace][UNK_TOKEN]
        else:
            return self._index2token[namespace][token_or_index]

    def size(self, namespace: str = "tokens") -> int:
        return len(self._token2index[namespace])
