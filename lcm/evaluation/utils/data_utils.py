# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import json
import logging
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice, zip_longest
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from fairseq2.data.data_pipeline import (
    DataPipeline,
    DataPipelineBuilder,
    read_iterator,
    read_sequence,
)
from fairseq2.gang import Gang
from fairseq2.typing import DataType
from jinja2 import Template

from lcm.datasets.configs import (
    ColumnsNames,
    DatasetConfig,
    DatasetConfigT,
    EvaluationDataLoadingConfig,
    JSONDatasetConfig,
    ParquetDatasetConfig,
    SonarTextColumn,
    create_dataset_config_from_cards,
)
from lcm.datasets.dataloading import build_parquet_iterator_pipeline
from lcm.evaluation.api import PREDICTION_TEXT_COLUMN, Example, Prompt, TaskConfig

source_key = ColumnsNames.source_column.value
source_text_key = ColumnsNames.source_text_column.value
target_key = ColumnsNames.target_text_column.value

logger = logging.getLogger(__name__)


@dataclass
class ResultsDatasetConfig(DatasetConfig):
    """The dataset that loads result from other evaluation task"""

    previous_result_dir: str = str()
    """Directory storing output of the previous task"""

    prompt_template: Optional[str] = None
    """
    A jinja-format string to apply for each item in the dataset to transform into a string.
    Useful for example when compiling a dynamic instruction / prompt for training or evaluation.
    Note that when this is specified, it will take precedence over the "affix" option, i.e. the
    columns `source_prefix_text`, `source_suffix_text`,... will be ignored.
    """


def renaming_keys(inp: Example, dataset: DatasetConfig) -> Example:
    mapper = {}
    for att in ColumnsNames.__members__:
        if att == ColumnsNames.dataset_name.value:
            continue
        if getattr(dataset, att, None):
            mapper[getattr(dataset, att)] = getattr(ColumnsNames, att).value

    dcols = dataset.columns
    if isinstance(inp, Dict):
        inp_cols = inp.keys()
        rename = {
            k: v
            for k, v in mapper.items()
            if k in inp_cols and (not dcols or k not in dcols)
        }
        append = {
            k: v for k, v in mapper.items() if k in inp_cols and (dcols and k in dcols)
        }

        for key, value in rename.items():
            inp[value] = inp.pop(key)
        for key, value in append.items():
            inp[value] = inp[key]

    elif isinstance(inp, pa.Table):
        inp_cols = inp.column_names
        rename = {
            k: v
            for k, v in mapper.items()
            if k in inp_cols and (not dcols or k not in dcols)
        }
        append = {
            k: v for k, v in mapper.items() if k in inp_cols and (dcols and k in dcols)
        }

        inp = inp.rename_columns([rename.get(key, key) for key in inp.column_names])
        for key, value in append.items():
            inp = inp.append_column(value, inp.column(key))

    else:
        raise NotImplementedError(
            f"Invalid type: {type(inp)} with no support for key renaming"
        )

    return inp


def load_jsonl(
    filename: str,
    num_shards: int = 1,
    shard_idx: int = 0,
    max_samples: Optional[int] = None,
) -> List[Example]:
    file = open(filename, "r", encoding="utf-8")
    with file as fh:
        lines = iter(fh)
        lines = islice(lines, shard_idx, max_samples, num_shards)
        return [json.loads(line) for line in lines]


def to_str(x: Union[str, list]) -> str:
    if isinstance(x, str):
        return x
    return " ".join(x)


def _add_affix(
    inp: Example,
    column: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    orig_column: str = None,
) -> Example:
    if column in inp.keys():
        inp[column] = str(prefix or "") + to_str(inp[column]) + str(suffix or "")
    else:
        assert orig_column in inp, (
            f"Missing {column} or {orig_column} (Found columns: {inp.keys()})"
        )
        inp[column] = str(prefix or "") + to_str(inp[orig_column]) + str(suffix or "")
    return inp


def _combine_sequences(
    inp: Example,
    column: str,
    sequences: List[SonarTextColumn],
) -> Example:
    texts = []
    for seq in sequences:
        if seq.text_value:
            texts.append(seq.text_value)
        else:
            assert seq.text_column
            texts.append(inp[seq.text_column])
    inp[column] = " ".join(texts)
    return inp


def jinja_format(x: Example, column: str, template: str):
    x[column] = Template(template).render(**x)
    return x


def _build_prompts_from_columns(pipe: DataPipelineBuilder, dataset: DatasetConfig):
    if getattr(dataset, "prompt_template", None):
        pipe.map(
            partial(
                jinja_format,
                column=source_text_key,
                template=getattr(dataset, "prompt_template"),
            )
        )
    elif dataset.source_sequences or dataset.target_sequences:
        if dataset.source_sequences:
            pipe.map(
                partial(
                    _combine_sequences,
                    column=source_text_key,
                    sequences=dataset.source_sequences,
                )
            )
        if dataset.target_sequences:
            pipe.map(
                partial(
                    _combine_sequences,
                    column=target_key,
                    sequences=dataset.target_sequences,
                )
            )
    else:
        # Add affixes before renaming the columns
        if dataset.source_text_column:
            pipe.map(
                partial(
                    _add_affix,
                    column=source_text_key,
                    prefix=dataset.source_prefix_text,
                    suffix=dataset.source_suffix_text,
                    orig_column=dataset.source_text_column,
                )
            )
        if dataset.target_text_column:
            pipe.map(
                partial(
                    _add_affix,
                    column=target_key,
                    prefix=dataset.target_prefix_text,
                    suffix=dataset.target_suffix_text,
                    orig_column=dataset.target_text_column,
                )
            )


def build_pipeline_from_jsonl(
    dataset: JSONDatasetConfig,
    loading_config: EvaluationDataLoadingConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataPipelineBuilder:
    examples = load_jsonl(
        filename=dataset.file_path,
        num_shards=world_size,
        shard_idx=rank,
        max_samples=loading_config.max_samples,
    )
    pipeline = read_sequence(examples)
    rename_func = partial(renaming_keys, dataset=dataset)
    pipeline.map(rename_func)
    _build_prompts_from_columns(pipeline, dataset)

    if loading_config.batch_size:
        pipeline.bucket(loading_config.batch_size)

    pipeline = pipeline.prefetch(int(loading_config.nb_prefetch * 5))

    if loading_config.max_iteration_steps is not None:
        pipeline = pipeline.take(loading_config.max_iteration_steps)

    return pipeline


def build_pipeline_from_parquet(
    dataset: ParquetDatasetConfig,
    loading_config: EvaluationDataLoadingConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataPipelineBuilder:
    pipeline = build_parquet_iterator_pipeline(
        dataset_config=dataset,
        loading_config=loading_config,
        rank=rank,
        world_size=world_size,
    )
    rename_func = partial(renaming_keys, dataset=dataset)
    pipeline.map(rename_func)

    return pipeline


def iterate_result_shard_files(result_dir: str) -> Iterator[Example]:
    """
    Iterate over the `previous_result_dir` and emit the tuple (json result, tensor result)
    in the ascending order of the shard indices. The shard files are assumed to
    follow pattern `*_<idx>.[json | pt]`
    """
    root_path = Path(result_dir)

    def get_shard_idx(fn: Path) -> int:
        _fn = fn.stem
        try:
            idx = int(_fn[_fn.rfind("_") + 1 :])
        except:  # noqa
            idx = 0
        return idx

    json_files = list(root_path.glob("raw_results/**/*.json*"))
    json_files = sorted(json_files, key=get_shard_idx)
    pt_files = list(root_path.glob("raw_results/**/*.pt"))
    pt_files = sorted(pt_files, key=get_shard_idx)

    for json_f, pt_f in zip_longest(json_files, pt_files):
        json_results = load_jsonl(str(json_f)) if json_f else []
        pt_ressults = torch.load(str(pt_f),map_location=torch.device('cpu')) if pt_f else []  # fmt: skip
        for json_result, pt_result in zip_longest(json_results, pt_ressults):
            example: Example = {}
            if json_result:
                example.update(json_result)
            if pt_result:
                example.update(pt_result)
            yield example


def build_pipeline_from_result_dir(
    dataset: ResultsDatasetConfig,
    loading_config: EvaluationDataLoadingConfig,
    rank: int = 0,  # Unused
    world_size: int = 1,
) -> DataPipelineBuilder:
    """The data loader for eval result files"""

    pipeline = read_iterator(
        iterator=iterate_result_shard_files(dataset.previous_result_dir),
        reset_fn=lambda _: iterate_result_shard_files(dataset.previous_result_dir),
        skip_pickling_check=True,
        infinite=False,
    )
    pipeline.shard(
        shard_idx=rank,
        num_shards=world_size,
        allow_uneven=not loading_config.even_sharding,
    )

    _build_prompts_from_columns(pipeline, dataset)
    if loading_config.batch_size:
        pipeline.bucket(loading_config.batch_size)

    pipeline = pipeline.prefetch(int(loading_config.nb_prefetch * 5))

    if loading_config.max_iteration_steps is not None:
        pipeline = pipeline.take(loading_config.max_iteration_steps)

    return pipeline


class EvalDataLoader(Generic[DatasetConfigT]):
    """The data loader API that works with lcm.evaluation.tasks"""

    dataset: DatasetConfigT
    data_config: EvaluationDataLoadingConfig

    def __init__(
        self,
        data_config: EvaluationDataLoadingConfig,
        dataset: DatasetConfigT,
        gang: Gang,
        builder_func: Callable[..., DataPipelineBuilder],
        dtype: DataType = torch.float16,
    ):
        self.data_config = data_config
        self.dataset = create_dataset_config_from_cards(dataset)  # type: ignore
        self.dtype = dtype
        self.gang = gang
        self.builder_func = builder_func

        self._pipeline: Optional[DataPipeline] = None

    @classmethod
    @abstractmethod
    def dataset_config(cls) -> Type[DatasetConfigT]:
        """
        Get the dataset type from the parameterized class.
        Surprisingly there is not a consistent way to get this from
        the Generics that can work with different Python versions,
        so we have to explicitly implement this in all sub-classes
        """
        ...

    @property
    def pipeline(self) -> DataPipeline:
        if self._pipeline is None:
            rank = getattr(self.data_config, "rank", 0)
            gang_rank = self.gang.rank if self.gang else 0
            if not rank and gang_rank:
                rank = gang_rank

            world_size = getattr(self.data_config, "world_size", 1)
            gang_world_size = self.gang.size if self.gang else 1
            if world_size == 1 and gang_world_size > 1:
                world_size = gang_world_size
            logger.info(
                f"Using rank={rank} among world_size={world_size} to build self._pipeline"
            )

            # In Evaluation data loader, max_samples can override the `max_iteration_steps`
            # and `batch_size` properties. Essentially
            if self.data_config.max_samples is not None:
                if not self.data_config.batch_size:
                    self.data_config.batch_size = (
                        self.data_config.max_samples // world_size
                    )
                batch_cnt = int(
                    self.data_config.max_samples
                    / (self.data_config.batch_size * world_size)  # fmt: skip
                )
                if rank == 0:
                    batch_cnt += int(self.data_config.max_samples % (self.data_config.batch_size * world_size) > 0)  # fmt: skip
                self.data_config.max_iteration_steps = batch_cnt

            self._pipeline = self.builder_func(
                self.dataset, self.data_config, rank, world_size
            ).and_return(max_num_warnings=4)

        return self._pipeline

    def iterate_batches(self) -> Iterator[List[Example]]:
        yield from self.pipeline


class JSONTestDataLoader(EvalDataLoader[JSONDatasetConfig]):
    def __init__(
        self,
        data_config: EvaluationDataLoadingConfig,
        dataset: JSONDatasetConfig,
        gang: Gang,
        dtype: DataType = torch.float16,
    ):
        super().__init__(
            data_config=data_config,
            dataset=dataset,
            gang=gang,
            builder_func=build_pipeline_from_jsonl,  # type: ignore
            dtype=dtype,
        )

    @classmethod
    def dataset_config(cls) -> Type[JSONDatasetConfig]:
        return JSONDatasetConfig


class ParquetTestDataLoader(EvalDataLoader[ParquetDatasetConfig]):
    """
    data loading that converts data into lcm.evaluation.api.Example items
    """

    def __init__(
        self,
        data_config: EvaluationDataLoadingConfig,
        dataset: ParquetDatasetConfig,
        gang: Gang,
        dtype: DataType = torch.float16,
    ):
        super().__init__(
            data_config=data_config,
            dataset=dataset,
            gang=gang,
            builder_func=build_pipeline_from_parquet,  # type: ignore
            dtype=dtype,
        )

    @classmethod
    def dataset_config(cls) -> Type[ParquetDatasetConfig]:
        return ParquetDatasetConfig

    def add_extra_columns(
        self, batch: Dict[str, Any], data_dict: Dict[str, Any]
    ) -> None:
        """Register additional columns in Example into the data_dict schema"""
        pass

    def iterate_batches(self) -> Iterator[List[Example]]:
        for batch in self.pipeline:
            source_inputs = []
            for x in batch[source_key]:
                if isinstance(x, torch.Tensor):
                    x = x.to(self.gang.device).to(self.dtype)
                source_inputs.append(x)
            data_dict = {}
            try:
                data_dict[target_key] = batch[target_key]
            except KeyError:
                pass
            try:
                data_dict[source_text_key] = list(map(as_py, batch[source_text_key]))
            except KeyError:
                pass

            # Add columns explicitly asked by the user
            _columns = self.dataset.columns or []
            for col in _columns:
                data_dict[col] = list(map(as_py, batch[col]))

            self.add_extra_columns(batch, data_dict)

            # We leave source column out of the pandas to keep them in cuda device

            data = pd.DataFrame(data_dict)
            examples = []
            for source_input, d in zip_longest(source_inputs, data.iterrows()):
                item = d[1].to_dict() if d is not None else {}
                item[source_key] = source_input
                examples.append(item)
            yield examples


class ResultsDataLoader(EvalDataLoader[ResultsDatasetConfig]):
    """
    A data loader that loads raw results of a generator (i.e. a directory with *.json
    and *.pt files)
    """

    prompt_template: Optional[str] = None
    """
    A jinja-format string to apply for each item in the dataset to transform into a string.
    Useful for example when compiling a dynamic instruction / prompt for training or evaluation.
    Note that when this is specified, it will take precedence over the "affix" option, i.e. the
    columns `source_prefix_text`, `source_suffix_text`,... will be ignored.
    """

    def __init__(
        self,
        data_config: EvaluationDataLoadingConfig,
        dataset: ResultsDatasetConfig,
        gang: Gang,
        dtype: DataType = torch.float16,
    ):
        super().__init__(
            data_config=data_config,
            dataset=dataset,
            gang=gang,
            builder_func=build_pipeline_from_result_dir,  # type: ignore
            dtype=dtype,
        )

    @classmethod
    def dataset_config(cls) -> Type[ResultsDatasetConfig]:
        return ResultsDatasetConfig


def as_py(obj: Any):
    if hasattr(obj, "as_py"):
        return obj.as_py()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def as_str(obj: Any, sep: str = " "):
    if isinstance(obj, list):
        return sep.join(obj)
    else:
        return obj


def is_tensor(obj: Any):
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, (np.ndarray, list, tuple)):
        if len(obj) == 0:
            return False
        return is_tensor(obj[0])
    return False


def is_lcm_input(x: Example) -> bool:
    """Check that an Example is a valid LCM input"""
    if source_key not in x:
        return False
    return isinstance(as_py(x[source_key]), torch.Tensor)


def default_embed_prompt(
    batch: Sequence[Example], config: TaskConfig
) -> Sequence[Prompt]:
    """
    A simply prompt that does not modify the input, just send its pre-sonarized embeddings out
    """
    if not is_lcm_input(batch[0]):
        raise NotImplementedError("Not support non-sonarized data")

    return [as_py(item[source_key]) for item in batch]


def default_text_prompt(batch: Sequence[Example]) -> Sequence[Prompt]:
    if source_text_key not in batch[0]:
        raise ValueError(
            f"Missing {source_text_key} in example (keys: {batch[0].keys()})"
        )

    return [as_str(item[source_text_key]) for item in batch]


def default_result_prompt(batch: Sequence[Example]) -> Sequence[Prompt]:
    prompts = []
    for item in batch:
        # IF the `source_text_key` is available, it takes precedence
        if source_text_key in item:
            prompts.append(to_str(item[source_text_key]))
        elif PREDICTION_TEXT_COLUMN in item:
            # item[PREDICTION_TEXT_COLUMN][0] --> the best prediction over multiple hypotheses
            prompts.append(to_str(item[PREDICTION_TEXT_COLUMN][0]))
        else:
            raise ValueError(
                f"Unsupport schema in result data loader: {item.keys()}. "
                "Expect a `_source_text_column` or `prediction_texts`"
            )
    return prompts


def default_text_postprocess(
    x: Example, source_text_column: Optional[str] = None
) -> Example:
    # Get the best hypothesis
    prediction_text = x[PREDICTION_TEXT_COLUMN][0]
    assert isinstance(prediction_text, str)

    res: Dict[str, Any] = {"prediction": prediction_text.strip()}

    if ColumnsNames.target_text_column.value in x:
        res["targets"] = [x[ColumnsNames.target_text_column.value]]

    if source_text_column in x:
        res["source"] = x[source_text_column]

    return res


def default_lcm_postprocess(
    x: Example, source_text_column: Optional[str] = None
) -> Example:
    # Get the best hypothesis
    prediction_text = x[PREDICTION_TEXT_COLUMN][0]
    assert isinstance(prediction_text, list), (
        f"LCM prediction texts are list of sentences, got {type(prediction_text)}"
    )

    preds = prediction_text

    res: Dict[str, Any] = {"prediction": " ".join(preds)}

    if ColumnsNames.target_text_column.value in x:
        targets = x[ColumnsNames.target_text_column.value]
        res["targets"] = [" ".join(targets)]

    if source_text_column in x:
        res["source"] = " ".join(x[source_text_column])

    return res
