import dataclasses
import json
import os
from pathlib import Path
from typing import ClassVar, Literal, Mapping

from ricode.ml.metrics import WordBoundarySpan
from ricode.ml.training_datasets import SplitInfo
from ricode.ml.training_utils import cached_property, map_if_not_none


def load_types_info(data_path: str, types_path: str):
    file_path = Path(data_path) / types_path
    if not file_path.exists():
        raise FileNotFoundError(file_path)

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
