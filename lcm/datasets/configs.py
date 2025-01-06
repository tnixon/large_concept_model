# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
import re
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

# XXX: these should be kept for eval of filters expressions
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fairseq2.assets import default_asset_store
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class ParquetBatchFormat(Enum):
    pyarrow = 0
    pandas = 1
    torch = 2


class ColumnsNames(Enum):
    source_column = "_source_column"
    source_text_column = "_source_text_column"
    target_column = "_target_column"
    target_text_column = "_target_text_column"

    dataset_name = "_dataset_name"


@dataclass
class SonarTextColumn:
    text_value: Optional[str] = None
    """
    Raw text expression that will be used as constant colum after being sententized and sonarized.
    """
    text_column: Optional[str] = None
    sonar_column: Optional[str] = None
    """
    Note `text_column` and `sonar_column` should be aligned (so `sonar_column` should be sonar encoded  `text_column`).
    If `sonar_column` is None and `text_column` is provided, we set `sonar_column = f"{text_column}_sonar_emb"` as default processing value!
    """


@dataclass
class ParquetDatasetLimitOptions:
    fraction_of_files: Optional[float] = None
    nb_files: Optional[int] = None
    nb_fragments: Optional[int] = None
    nb_rows: Optional[int] = None


@dataclass(frozen=True)
class SonarDecoderConfig:
    tokenizer: str = "text_sonar_basic_decoder"
    """ SONAR tokenizer """

    decoder: str = "text_sonar_basic_decoder"
    """ SONAR decoder"""

    lang: str = "eng_Latn"
    """ Target language """

    max_tokens_in_sentence: int = 256
    """Maximum number of tokens generated in the text"""

    temperature: float = 1.0
    """The decoding logit temperature, where values greater than 1.0 produce more
        uniform logits; values less than 1.0 produce sharper logits."""


@dataclass(frozen=True)
class SonarEncoderConfig:
    tokenizer: str = "text_sonar_basic_encoder"
    """ SONAR tokenizer """

    encoder: str = "text_sonar_basic_encoder"
    """ SONAR decoder"""

    lang: str = "eng_Latn"
    """ Target language """


@dataclass
class DatasetConfig:
    """
    Generic dataset config
    """

    columns: Optional[List[str]] = None
    """The list of columns to load.
    Columns such as `source_column`, ..., will be added automatically.
    """

    source_text_column: Optional[str] = None
    """ Column to load as source raw text"""

    target_text_column: Optional[str] = None
    """ Column to load as target raw text for paired data"""

    source_prefix_text: Optional[str] = None
    """ Text to prepend to the content of the source_column"""

    source_suffix_text: Optional[str] = None
    """ Text to append to the content of the target_column"""

    target_prefix_text: Optional[str] = None
    """ Text to prepend to the content of the source_column"""

    target_suffix_text: Optional[str] = None
    """ Text to append to the content of the target_column"""

    source_sequences: Optional[List[SonarTextColumn]] = None
    """
    Designed to make on-the-fly prompts from existing columns that are more complex than prefix and suffix.
    Each element of source_sequences  is a SonarTextColumn, which can be either:
    - constant raw text (with the text_value argument)
    - text column (with the text_column argument)
    - sonar column (with the sonar_column argument)

    Note that text_value cannot co-exist with text_column or sonar_column, and sonar column cannot be specified
    without a text column. Further behaviour for parquet datasets:
    - If text_value is specified, this  will be split to sentences and sonarized
    - If only text_column is specified, a new column named "<text_column>_sonar_emb" will be added as sonar_column.
    - If both (text_column, sonar_column) is specified,

    All SonarTextColumn elements from source_sequences will be concatenated together to produce new source_column
    and source_text_column (same for target), which will have names as defined in ColumnsNames.
    Using source_sequences is NOT compatible with using source_column or source_text_column, as well as quality filtering.
    """

    target_sequences: Optional[List[SonarTextColumn]] = None
    """Designed to make on-the-fly prompts / instructions for target column, see `source_sequences` for more details"""

    silent_freeze: bool = False
    """If set to true, the config value can only be set once, i.e. it will not be able to update after the being set is instantiated.
    This is helpful to avoid side-effect in setting some configs after being specified by the user application (Hydra, CLI)"""

    def __post_init__(self):
        if self.source_sequences is not None:
            if self.source_text_column is not None:
                logger.warning(
                    f"Both `source_sequence` and `source_text_column` is specified. "
                    f"Ignore `source_text_column` and use default value `{ColumnsNames.source_text_column.value}`.\n"
                    f"(`source_sequences` = {self.source_sequences}, \n"
                    f"`source_text_column` = {self.source_text_column} )"
                )
            self.source_text_column = ColumnsNames.source_text_column.value

        if self.target_sequences is not None:
            if self.target_text_column is not None:
                logger.warning(
                    f"Both `target_sequences` and `target_text_column` is specified. "
                    f"Ignore `target_text_column` and use default value `{ColumnsNames.target_text_column.value}`.\n"
                    f"(`target_sequences` = {self.target_sequences}, \n"
                    f"`target_text_column` = {self.target_text_column} )"
                )
            self.target_text_column = ColumnsNames.target_text_column.value

        for col in (self.source_sequences or []) + (self.target_sequences or []):
            if col.text_value is not None:
                assert col.text_column is None and col.sonar_column is None
            else:
                assert col.text_column is not None

        self._has_initialized_: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_has_initialized_", False):
            return super().__setattr__(name, value)
        if name == "silent_freeze":
            raise ValueError(
                "Direct change of silent_freeze outside __init__ is forbidden"
            )
        if self.silent_freeze and getattr(self, name) not in ("", None, MISSING):
            logger.debug(
                f"Ignore change of {name} since silent_freeze is set and value is not empty ({getattr(self, name)})"
            )
            return
        super().__setattr__(name, value)

    def override_attr(self, name: str, value: Any) -> None:
        try:
            self._has_initialized_ = False
            super().__setattr__(name, value)
        finally:
            self._has_initialized_ = True

    def freeze(self) -> None:
        """Turn the `silent_freeze` flag on"""
        try:
            self._has_initialized_ = False
            self.silent_freeze = True
        finally:
            self._has_initialized_ = True


@dataclass
class JSONDatasetConfig(DatasetConfig):
    """Config for datasets stored in JsonL format."""

    file_path: str = str()
    """
    Path to the directory containing the Jsonl dataset.
    Each task will replace this wil a real Json files
    TODO: Add support for remote JsonL file (e.g. with "s3://...")
    """

    prompt_template: Optional[str] = None
    """
    A jinja-format string to apply for each item in the dataset to transform into a string.
    Useful for example when compiling a dynamic instruction / prompt for training or evaluation.
    Note that when this is specified, it will take precedence over the "affix" option, i.e. the
    columns `source_prefix_text`, `source_suffix_text`,... will be ignored.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if not getattr(self, "_has_initialized_", False):
            return super().__setattr__(name, value)

        if name == "silent_freeze":
            raise ValueError("Direct change of silent_freeze is forbidden")

        if self.silent_freeze:
            if getattr(self, name) not in ("", None, MISSING):
                logger.debug(
                    f"Ignore change of {name} in silent frozen mode when value is not empty ({getattr(self, name)})"
                )
                return

            # Ensure we cannot set the default `prompt_template` value when the user specifies
            # source_sequences or source_text_column explicitly
            for hi_prior_col, lo_prior_col, lo_prior_value in [
                ("source_sequences", "source_text_column", self.source_text_column),
                ("target_sequences", "target_text_column", self.target_text_column),
                ("prompt_template", "source_sequences", self.source_sequences),
                ("prompt_template", "source_prefix_text", self.source_prefix_text),
                ("prompt_template", "source_suffix_text", self.source_suffix_text),
            ]:
                if name == hi_prior_col and lo_prior_value not in ("", None, MISSING):
                    logger.warning(
                        f"Updating value of {hi_prior_col} will cause conflicts with the user-defined "
                        f"value in {lo_prior_col}. The update will be ignored.\n"
                    )
                    return

        super().__setattr__(name, value)


@dataclass
class ParquetDatasetConfig(DatasetConfig):
    """
    Config for datasets stored in Parquet format.

    XXX: this config should not hold non-trival default values.
    We want this to make datacards info and hydra config merge easier.
    All None value should be filled up in downstream `build_parquet_iterator_pipeline`.
    """

    name: Optional[str] = None
    """When name is provided, it will use preregistered cards to populate all attributes.
        name convention is the following
        -  {card_name}={split}:{weight}

       Example:
        - wiki
        - wiki:0.2 # no split
        - wiki=dev  # default weight=1
        - wiki=dev:0.2

       Cards attributes will be overwritten by user defined ParquetDatasetConfig in
            `create_dataset_config_from_cards`.
    """

    parquet_path: str = str()
    """The path to parquet dataset file.
        if `parquet_path` is remote (like stats with "s3://..."),
        the filesystem will be automatically detected and `filesystem_expr` should remain None
    """

    weight: float = 1.0
    """
    Indicates relative weight of dataset that can be used for sampling from different datasets.
    """

    limit: Optional[ParquetDatasetLimitOptions] = None
    """
    Contains different options that allows to load only a part of the provided dataset.
    It will **always** take some number of **first** fragments according to the order in which
    they appear in the dataset and this logic will not be depedent on suffling/seed.
    When several limits are provided, each of them will be applied (resulting in the strongest limit).
    """

    source_column: Optional[str] = None
    """ Column to load as source embeddings"""

    target_column: Optional[str] = None
    """ Column to load as target embeddings for paired data"""

    source_quality_column: Optional[str] = None
    source_quality_range: Optional[Any] = None

    partition_filters: Optional[str] = None
    """
    Filters that should be applied only on partition columns for fast partition prunning.
    This filters should not be duplicated in `filters` (below) which are used on materialized data.
    To know the partition columns on dataset :
    ```python
    >>> pq.ParquetDataset(parquet_path).partitioning.schema.names
    ```
    Note that for if `parquet_path` references a single file -> the result above will NOT be correct (returns all columns).
    Note that for a single file case, there should no partition_filters since there're no partitions !!
    """

    filters: Optional[str] = None
    """See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression

    Some examples :

    >>> import pyarrow.compute as pc
    >>> import pyarrow as pa

    >>> filters = (pc.field("data_split") == pc.scalar("train")) & (pc.field("duration") > 7)
    >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    >>> filters = pa.compute.less_equal(pa.compute.list_value_length(pa.dataset.field("audio_wav")), 16_000 * 30)

    Note that all fields used here should be among existing columns in the dataset schema.
    For hydra compatibility, we need to pass this filters as an str expression that'll be passed to `eval(...)`
    """

    filesystem_expr: Optional[str] = None
    """
    DEPRECATED : not used any more and will be remove soon
    """

    filesystem: Optional[Any] = None
    """
    DEPRECATED: not used any more and will be remove soon
    """

    split_to_row_groups: Optional[bool] = None
    """If ``True``, uses Parquet row groups instead of simple partitions which
    are generally smaller. Highly recommended for non-partitioned parquet files."""

    nb_parallel_fragments: Optional[int] = None
    """
    This parameter can be dataset specific:
    For dataset with large number of sentences per document (sample),
    it's enough to set `nb_parallel_fragments=2 or 3`.
    For datasets, with smaller number of sentences (~10) and small row_group_size (~200-600),
     `nb_parallel_fragments` could be increase to 10 - 20.

    The number of Parquet fragments allowed to be read in parallel. Higher
    values will result in higher speeds, better randomization, and higher memory
    footprint. If partition size is rather small compared to the batch size, we
    recommend to increase ``nb_parallel_fragments``.

    Leaving ``nb_parallel_fragments`` to None will trigger auto-detection based on dataset metadata.
    """

    sharding_in_memory: bool = False
    """
    This option should be activated for sharding small datasets whose total number of row groups is small
    that makes sharding per row group impossible.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.source_sequences is not None:
            if self.source_column is not None:
                logger.warning(
                    f"Both `source_sequences` and `source_column` is specified. "
                    f"Ignore `source_column` and use default value `{ColumnsNames.source_column.value}`.\n"
                    f"(`source_sequences` = {self.source_sequences}, \n"
                    f"`source_column` = {self.source_column} )"
                )
            assert self.source_quality_range is None
            self.source_column = ColumnsNames.source_column.value

        if self.target_sequences is not None:
            if self.target_column is not None:
                logger.warning(
                    f"Both `target_sequences` and `target_column` is specified. "
                    f"Ignore `target_column` and use default value `{ColumnsNames.target_column.value}`.\n"
                    f"(`target_sequences` = {self.target_sequences}, \n"
                    f"`target_column` = {self.target_column} )"
                )
            self.target_column = ColumnsNames.target_column.value

        for col in (self.source_sequences or []) + (self.target_sequences or []):
            if col.sonar_column is None and col.text_value is None:
                assert col.text_column, f"Invalid SonarTextColumn: {col}"
                col.sonar_column = col.text_column + "_sonar_emb"

        if self.source_quality_range is None:
            self.source_quality_column = None


DatasetConfigT = TypeVar("DatasetConfigT", bound=DatasetConfig, contravariant=True)


@dataclass
class DataLoadingConfig:
    multiple_dataset_chaining: str = "sample"
    """
    This option allows to chain several datasets together.
    The chaining can be done in two ways:
    - `sample` : each dataset will be sampled with the provided weight
    - `concat` : datasets will be concatenated together (no weights taken into account)
    - `round_robin`: datasets will be sampled in a round robin fashion (no weights taken into account)
    """
    batch_size: Optional[int] = None
    """The output batch size."""

    order_by_length: bool = True
    """
    Whether to create the batches with homogeneous tokens length
    for more efficient padding.
    """

    max_tokens: Optional[int] = None
    """Used with the ``order_by_length`` option to control the total number of
    padded tokens in each batch. Typically, this option is preferred over
    ``batch_size`` to reduce the memory footprint.
    """

    len_to_wrap_long_seq: Optional[int] = None
    """
    Wrapping a source sequences to the length of `len_to_wrap_long_seq`.
    For instance, for a `len_to_wrap_long_seq=2`
    batch = {
        "source": [["v1", "v2", "v3", "v4", "v5"], ["u1", "u2", "u3"], ["w1"]],
    }
    will be transormed to
    1. if packing is False :
    batch = {
        "source": [['v1', 'v2'], ['v3', 'v4'], ['v5'], ["u1", "u2"], ["u3"], ["w1"]]
    }
    1. if packing is True :
    batch = {
        "source": [['v1', 'v2'], ['v3', 'v4'], ['v5', 'u1'], ["u2", "u3"], ["w1"]]
    }

    Note: currently only allowed to be used with no "target" provided (unsupervised style) !
    """

    packing: bool = False
    """
    If True, all sequential documents (seqs of sentences) will be concated into one big document
    before applying wrapping.
    This will result in all samples (except maybe one) having exactly `len_to_wrap_long_seq` length !
    """

    wrap_before_affixing: bool = False
    """
    If True, we will wrap the sequences before adding the source prefix/suffix.
    Recommended when pre-training with packed data i.e len_to_wrap_long_seq not None and packing=True
    """

    max_sentence_len_in_doc: Optional[int] = None
    """
    Remove samples (documents) whose `source_text_column` contains at least one sentence of len > `max_sentence_len_in_doc`.
    This operations is done after long sequences wrapping (if applicable).
    Typically values:  100 - 300
    """
    min_sentence_len_in_doc: Optional[int] = None
    """
    Remove samples (documents) `source_text_column` contains at least one sentence of len < `min_sentence_len_in_doc`.
    This operations is done after long sequences wrapping (if applicable).
    Typically values:  5 - 15
    """

    max_sentence_len_in_target_doc: Optional[int] = None
    """
    same filtering option as above but for `target_text_column`
    """
    min_sentence_len_in_target_doc: Optional[int] = None
    """
    same filtering option as above but for `target_text_column`
    """

    min_length_of_sequences: Optional[int] = 1
    """
    Remove samples (documents) whose `source_text_column` are scrictly shorter than `min_length_of_sequences`.
    This operations is done after long sequences wrapping (if applicable).
    One can use here the same value as for sequences wrapping
    in order to produce all sequences with the same length.
    """
    min_length_of_sequences_after_batching: Optional[int] = 1
    """
    Remove source sequences shorter than `min_length_of_sequences_after_batching`
    This filtering is applied after batching and potentially affixing and wrapping.
    """
    min_length_of_target_sequences: Optional[int] = 1
    """
    Same as above applied for `target_text_column`
    """
    min_length_of_target_sequences_after_batching: Optional[int] = 1
    """
    Same as above applied for `target_text_column`
    """

    output_format: ParquetBatchFormat = ParquetBatchFormat.torch
    """The format to use for output batches."""

    shuffle: bool = True
    """If ``True``, shuffles the dataset samples during the iteration. If ``False``
    and ``order_by_length`` is ``None``, the batch samples will be produced in
    natural Parquet dataset reading order."""

    drop_null: bool = True
    """If ``True``, drops rows containing any null value."""

    seed: int = 123
    """The RNG seed value for deterministic behavior."""

    nb_epochs: int = 100
    """
    Number of passes over the data before iterations stop
    """

    min_batch_size: int = 1
    """Drops batches whose length is less than ``min_batch_size``"""

    nb_prefetch: float = 3.0
    """The number of producer groups (of size `nb_parallel_fragments`) to
    prefetch."""

    num_parallel_calls: float = 1.5
    """The number of parallel calls in map operations."""

    use_threads: bool = False
    """Whether pyarrow should use its internal threads to read the Parquet file.
    Since we rely on the external parallelism, this param is tuned off by
    default."""

    ignore_checkpointed_pipeline: bool = False
    """Whether to ignore the saved datapipeline state or load it when resuming.
    Temporary fix for issues re-loading saved checkpoints"""

    even_sharding: bool = False
    """
    This option should be activated ONLY for validataion on small datasets
    to guarantee the perfect data sharding accross the workers.
    Note that in current impmentation, activating `even_sharding` requires `sharding_in_memory=True`
    which will lead to big overhead for big dataset.
    Note also that some fraction of the data may be dropped due to even sharding.
    For big validation datasets, prefer using large `nb_epoch` + limiting `max_validation_iterations`
    instead of using `even_sharding` !

    For training use case, it should left to False and combined with large number of epochs.
    For evaluation use case, it also should be False since we dont care about the batch syncronization across different workers.
    """
    max_iteration_steps: Optional[int] = None
    """
    If not None, it will be used to limit the number of batches produced per each dataset
    """


@dataclass
class ValidationDataLoadingConfig(DataLoadingConfig):
    """
    This class allows to have some hardcoded parameters for data loading of validation datasets
    """

    multiple_dataset_chaining: str = "concat"
    nb_epochs: int = 1
    min_batch_size: int = 1  # we want to keep all samples
    shuffle: bool = False  # we dont need the randomness here
    batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    """
    Leaving both `max_tokens` and `batch_size` to None will trigger auto-detection based on dataset metadata and distributed training world size.
    to make more or less even distribution of samples across workers. Typically,
    if worker_batch_size = total_batch_size // world_size <= 40, we will use batch_size=worker_batch_size,
    otherwise we will use max_tokens=min(total_tokens_number // world_size, 3000).
    See dataloading:SingleParquetDatasetDataloader::set_validation_params for more details.
    """


@dataclass
class EvaluationDataLoadingConfig(DataLoadingConfig):
    """
    This class allows to have some hardcoded parameters for data loading of evaluation datasets.
    In partitcular, even in distributed setup evaluation should not require workers syncronization.
    Therefore, we set `even_sharding` = False to get the all data samples !
    """

    multiple_dataset_chaining: str = "concat"
    nb_epochs: int = 1  # only ONE full pass over the full data !
    min_batch_size: int = 1  # we want to keep all samples
    shuffle: bool = False  # we dont need the randomness here
    batch_size: Optional[int] = 10
    max_tokens: Optional[int] = None  # this should be ok for most of models
    even_sharding: bool = False  # we dont want to lose any sample !
    sharding_in_memory: bool = True  # activate sharding by rank and world size
    rank: int = 0
    world_size: int = 1
    max_samples: Optional[int] = None  # fmt: skip
    """evaluate only the first n samples (for debugging)"""


def setup_fairseq2_extensions() -> None:
    # path where all datacards should be located !
    cards_dir = Path(__file__).parent.parent.joinpath("datacards")
    if cards_dir.exists():
        default_asset_store.add_file_metadata_provider(cards_dir)


setup_fairseq2_extensions()


def get_cluster() -> Optional[str]:
    """Returns the cluster name of the current environment.
    User can implement their own logic to load datasets living in different locations/clusters
    """
    return "s3"


def _resolve_parquet_path(options: Dict[str, str]) -> Optional[str]:
    cluster_name = get_cluster() or "s3"

    parquet_path = options.get(cluster_name)
    if parquet_path is None:
        # best effort - taking first element
        parquet_path = next(iter(options.values()))

    return parquet_path


def _resolve_filters(
    split: Optional[str],
    card_filter: Optional[str],
    user_filter: Optional[str],
    card_partition_filters: Optional[str],
    user_partition_filters: Optional[str],
) -> Tuple[Optional[pc.Expression], Optional[pc.Expression]]:
    custom_filters = user_filter or card_filter
    partition_filters = user_partition_filters or card_partition_filters

    if custom_filters is not None:
        custom_filters = pq.filters_to_expression(eval(custom_filters))

    if partition_filters is not None:
        partition_filters = pq.filters_to_expression(eval(partition_filters))

    if split:
        split_filter = pc.equal(pc.field("split"), split)
        if partition_filters is None:
            partition_filters = split_filter
        else:
            partition_filters = pa.compute.if_else(
                split_filter, partition_filters, False
            )

    return custom_filters, partition_filters


def _default_resolver(a, b):
    res = a if bool(a) and a is not MISSING else b
    return res


def get_parquet_config_from_name(
    name: str, config: Optional[ParquetDatasetConfig] = None
) -> ParquetDatasetConfig:
    """
    name convention is the following
    -  {card_name}={split}:{weight}
    """
    # parsing name
    pattern = r"^(?P<card_name>[a-zA-Z0-9_]+)=?(?P<split>[a-zA-Z0-9_]*)?:?(?P<weight>\d+(?:\.\d+)?)?$"
    match_ = re.match(pattern, name)
    assert match_ is not None, f"name parsing failed: {name}"
    card_name = match_.group("card_name")
    split = match_.group("split")
    weight = match_.group("weight")

    if weight:
        weight = float(weight)
    logger.info(
        f"Parsing {name} : card_name={card_name}, split={split}, weight={weight}"
    )

    reload_config = default_asset_store.retrieve_card(card_name)
    cards_metadata: Dict[str, Any] = {**reload_config._metadata}

    if config is None:
        config = ParquetDatasetConfig(name=card_name, parquet_path="")

    assert config is not None

    if isinstance(config, ParquetDatasetConfig):
        config_dict = asdict(config)
    else:
        config_dict = config  # type: ignore

    metadata = {}
    # resolve parquet_path according to the cluster
    for field in fields(ParquetDatasetConfig):
        field_name = field.name
        metadata[field_name] = _default_resolver(
            config_dict.get(field_name), cards_metadata.get(field_name)
        )

    if isinstance(metadata["source_sequences"], list):
        metadata["source_sequences"] = [
            SonarTextColumn(**item) for item in metadata["source_sequences"]
        ]

    if isinstance(metadata["target_sequences"], list):
        metadata["target_sequences"] = [
            SonarTextColumn(**item) for item in metadata["target_sequences"]
        ]

    metadata["parquet_path"] = _default_resolver(
        config_dict.get("parquet_path"),
        _resolve_parquet_path(cards_metadata["parquet_path"]),
    )

    metadata["filters"], metadata["partition_filters"] = _resolve_filters(
        split,
        card_filter=cards_metadata.get("filters"),
        user_filter=config_dict.get("filters"),
        card_partition_filters=cards_metadata.get("partition_filters"),
        user_partition_filters=config_dict.get("partition_filters"),
    )
    if weight:  # priority from parsed name
        metadata["weight"] = weight
    metadata["name"] = name

    # to patch nested hydra case !
    if metadata["limit"] is not None and isinstance(metadata["limit"], dict):
        metadata["limit"] = ParquetDatasetLimitOptions(**metadata["limit"])

    return ParquetDatasetConfig(**metadata)


def create_dataset_config_from_cards(
    config: DatasetConfig,
) -> DatasetConfig:
    if getattr(config, "name", None) is None:
        return config
    output_config = get_parquet_config_from_name(config.name, config)  # type: ignore
    return output_config


def get_renaming_mappers(configs: Sequence[DatasetConfig]) -> List[dict]:
    used_columns = [x for x in ColumnsNames.__members__ if x != "dataset_name"]

    pre_mapping = {
        att: [getattr(cc, att) for cc in configs if hasattr(cc, att)]
        for att in used_columns
    }

    mappers: List[dict] = [{} for _ in configs]
    for att, val in pre_mapping.items():
        if all(x is None for x in val):
            continue
        for i, name in enumerate(val):
            if name is None:
                raise ValueError(
                    f"All datasets should provide {att} param, but got {configs[i]}"
                )
            mappers[i][name] = getattr(ColumnsNames, att).value
    return mappers
