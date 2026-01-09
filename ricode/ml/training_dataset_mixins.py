import dataclasses
import json
import os
import warnings
from pathlib import Path
from typing import ClassVar, Literal, Mapping, Optional

from safetensors_dataset import SafetensorsDataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from ricode.ml.metrics import WordBoundarySpan
from ricode.ml.training_datasets import BasicDataset, SplitInfo
from ricode.ml.training_utils import cached_property, map_if_not_none


def load_types_info(data_path: str, types_path: str):
    file_path = Path(data_path) / types_path
    if not file_path.exists():
        alternative_filepath = Path("..") / types_path
        if alternative_filepath.exists():
            file_path = alternative_filepath
        else:
            raise FileExistsError

    with open(file_path) as types_f:
        types_info = json.load(types_f)

    info = TypesInfo(
        list(sorted(types_info["entities"].keys())),
        map_if_not_none(
            types_info.get("relations", None), lambda t: list(sorted(t.keys()))
        ),
        map_if_not_none(
            types_info.get("relations", None),
            lambda t: [k for k, v in t.items() if v.get("symmetric", False)],
        ),
        type=types_info.get(
            "type",
            (
                "ner"
                if len(types_info.get("relations", [])) == 0
                else "relation_extraction"
            ),
        ),
        nest_depth=types_info.get("nest_depth", 1),
    )
    return info


class TokenizerMixin:
    tokenizer_class: ClassVar[PreTrainedTokenizerBase] = AutoTokenizer

    model_max_length: int
    additional_special_tokens: Optional[list[str]] = None

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        tk = self.__dict__.get("tokenizer")
        if isinstance(tk, PreTrainedTokenizerBase):
            return tk

        path = self.__dict__.get("_tokenizer_config_path")
        if path:
            return self.setup_tokenizer(path)
        raise AttributeError("Tokenizer wasn't set as path or object")

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase | str):
        if isinstance(tokenizer, str):
            self.__dict__["_tokenizer_config_path"] = tokenizer
        elif isinstance(tokenizer, PreTrainedTokenizerBase):
            self.__dict__["tokenizer"] = tokenizer
        else:
            raise ValueError("Tokenizer didn't have the expected types")

    @cached_property
    def tokenizer_config(self) -> PretrainedConfig:
        try:
            return AutoConfig.from_pretrained(self.tokenizer.name_or_path)
        except ValueError as cause:
            warnings.warn(
                f"Caught {cause} when trying to fetch config for {self.tokenizer.name_or_path}"
            )
            raise AttributeError

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length

    def setup_tokenizer(
        self, pretrained_model_name_or_path: str
    ) -> PreTrainedTokenizerBase:
        return self.tokenizer_class.from_pretrained(
            pretrained_model_name_or_path,
            model_max_length=self.model_max_length,
            additional_special_tokens=self.additional_special_tokens,
        )


@dataclasses.dataclass(kw_only=True, repr=False)
class TokenizerDataset(BasicDataset, TokenizerMixin):
    tokens_key: ClassVar[str] = "tokens"
    tokenize_kwargs: ClassVar[Optional[dict]] = None
    do_bulk_tokenize: ClassVar[bool] = False

    model_max_length: int
    tokenizer: PreTrainedTokenizerBase

    def json_list_setup_examples(
        self, split: str, split_info: SplitInfo, json: list[dict]
    ) -> SafetensorsDataset:
        if self.do_bulk_tokenize:
            tokens_list = list()
            is_split_into_words = False
            is_generator = not isinstance(json, list)
            elements = list()
            for elem in tqdm(json, leave=True):
                if is_generator:
                    elements.append(elem)
                tokens = elem[self.tokens_key]
                if not isinstance(tokens, (str, list)):
                    raise ValueError(
                        f"tokens must be a str or list, got {type(tokens)}"
                    )
                tokens_list.append(tokens)
                if not is_split_into_words and isinstance(tokens, list):
                    is_split_into_words = True
            if is_generator:
                json = elements

            tokenize_kwargs = self.tokenize_kwargs or dict()
            batch_encoding = self.tokenizer(
                tokens_list, is_split_into_words=is_split_into_words, **tokenize_kwargs
            )

            for pos, elem in enumerate(tqdm(json, leave=True)):
                for key, value in batch_encoding.items():
                    if key in elem:
                        raise ValueError(
                            f"elements of loaded dataset cannot define {key}: {elem.keys()}"
                        )
                    elem[key] = value[pos]
                if (
                    "return_offsets_mapping" in tokenize_kwargs
                    and tokenize_kwargs["return_offsets_mapping"]
                ):
                    elem["word_ids"] = batch_encoding.word_ids(pos)
        return super().json_list_setup_examples(split, split_info, json)


class ContaminatedMixin:
    name: str
    splits: Mapping[str, SplitInfo | dict[str, SplitInfo]]

    @property
    def data_path(self) -> Path:
        raise NotImplementedError

    @cached_property
    def contaminated_train_entities(self) -> set[WordBoundarySpan]:
        contaminated_split = self.splits.get(
            "contaminated", SplitInfo(name_or_file=f"{self.name}_contaminated.json")
        )
        if not isinstance(contaminated_split, SplitInfo):
            raise ValueError
        contaminated_path = os.path.join(
            "", self.data_path, contaminated_split.name_or_file
        )
        if not os.path.exists(contaminated_path):
            setattr(self, "contaminated_train_entities", set())
            return set()

        with open(contaminated_path) as contaminated_f:
            contaminated_info = json.load(contaminated_f)
        contaminated_train = contaminated_info.get("train", None)
        entities = {
            WordBoundarySpan(
                (entity["start"], entity["end"]), entity["type"], tuple(entity["words"])
            )
            for entity in contaminated_train
        }
        setattr(self, "contaminated_train_entities", entities)
        return entities


@dataclasses.dataclass()
class TypesInfo:
    entity_types: list[str]
    relation_types: list[str] | None
    symmetric_relation_types: list[str] | None
    type: Literal["ner", "relation_extraction", "entity_linking"]
    nest_depth: int

    @property
    def num_entity_types(self):
        return len(self.entity_types)

    @property
    def num_relation_types(self):
        if self.relation_types is None:
            return 0
        return len(self.relation_types)


class TypesMixin:
    with_links: ClassVar[bool] = False

    @cached_property
    def entity_types(self) -> list[str]:
        return self.types_info.entity_types

    @cached_property
    def link_types(self) -> list[str]:
        if not self.with_links:
            raise ValueError(
                f"Not loading relation information when {self.with_links=}"
            )
        elif self.types_info.relation_types is None:
            raise ValueError
        return self.types_info.relation_types

    @cached_property
    def symmetric_link_types(self) -> set[str]:
        if not self.with_links:
            raise ValueError(
                f"Not loading relation information when {self.with_links=}"
            )
        elif self.types_info.symmetric_relation_types is None:
            raise ValueError
        return self.types_info.symmetric_relation_types

    @cached_property
    def num_types(self) -> int:
        return len(self.entity_types)

    @cached_property
    def num_links(self) -> int:
        return len(self.link_types)

    @cached_property
    def types_info(self) -> TypesInfo:
        return self._resolve_types_using_splits()

    def _resolve_types_using_splits(self):
        if not hasattr(self, "splits") or not isinstance(getattr(self, "splits"), dict):
            raise ValueError(f"{self.__class__.__name__}.splits must be a dict")

        splits: dict = getattr(self, "splits")

        if "types" not in splits:
            raise ValueError(f'Cannot find key "types" in splits, got {splits.keys()}')
        types_path = splits["types"]
        return self._resolve_types_file(types_path.name_or_file)

    def _resolve_types_file(
        self,
        types_path: str,
    ) -> TypesInfo:
        if not hasattr(self, "data_path") or not isinstance(
            getattr(self, "data_path"), Path
        ):
            raise ValueError(f"{self.__class__.__name__}.data_path must be a Path")
        return load_types_info(
            getattr(self, "data_path").as_posix(),
            types_path,
        )
