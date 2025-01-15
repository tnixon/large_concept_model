# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from typing import Any, Generator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder
from fairseq2.data.parquet.tools import BatchOutputType, apply_filter, concat_table
from pyarrow.dataset import get_partition_keys
from stopes.utils.arrow_utils import (
    explode_table_with_fixed_length,
    explode_table_with_max_length,
    is_list_like,
)

from lcm.datasets.configs import (
    DataLoadingConfig,
    ParquetBatchFormat,
    ParquetDatasetConfig,
    ValidationDataLoadingConfig,
    get_renaming_mappers,
)
from lcm.datasets.parquet_utils import (
    build_batching_loop_over_one_table,
    define_parquet_dataset,
    filter_document_by_quality,
    filter_long_short_sentence_document,
    filter_table_with_different_lengths,
    get_row_group_level_metadata,
    materialize_sequence,
    prefix_and_suffix_one_list_column,
    prepare_suffix_prefix_embeddings,
    pyarrow_table_to_torch_dict,
    renaming,
    shuffle_table,
    stream_parquet_fragments,
)

logger = logging.getLogger(__name__)

PA_NB_CPU = 4
pa.set_cpu_count(PA_NB_CPU)
pa.set_io_thread_count(PA_NB_CPU)


def return_none_on_failure(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    return wrapper


@dataclass
class GlobalPQStats:
    min_number_of_fragment: int
    mean_fragment_length: float
    mean_fragment_number_of_tokens: Optional[float] = None


class SingleParquetDatasetDataloader:
    _pq_ds: Optional[pq.ParquetDataset] = None
    proxy_number_of_fragments: int
    basic_stats: GlobalPQStats

    def __init__(
        self, dataset_config: ParquetDatasetConfig, loading_config: DataLoadingConfig
    ):
        self.dataset_config = deepcopy(dataset_config)
        self.loading_config = deepcopy(loading_config)
        self.config_post_init()
        nb_parallel_fragments = self.dataset_config.nb_parallel_fragments
        assert isinstance(nb_parallel_fragments, int)
        self.nb_parallel_fragments: int = nb_parallel_fragments

    @property
    def is_validation(self) -> bool:
        return isinstance(self.loading_config, ValidationDataLoadingConfig)

    def head(self, top=5):
        return self.dataset._dataset.head(top)

    @property
    def dataset(self) -> pq.ParquetDataset:
        if self._pq_ds is None:
            self._pq_ds = self.get_dataset()

        return self._pq_ds

    @property
    def full_schema(self) -> pa.Schema:
        return self.dataset.schema

    def _warn_filters_usage(self, pq_ds: pq.ParquetDataset) -> None:
        partition_filters = self.dataset_config.partition_filters

        frags = pq_ds.fragments
        if len(frags) == 0:
            raise ValueError(
                f"Working on empty dataset, probably due to wrong `partition_filters` definition : {partition_filters}"
            )

        partition_columns = list(
            get_partition_keys(frags[0].partition_expression).keys()
        )
        if not partition_columns and partition_filters is not None:
            raise ValueError(
                f"Partition filters {partition_filters} is set but dataset has NO partition columns"
            )

        if partition_columns and partition_filters is not None:
            expression_candidates = [
                x for x in partition_columns if x in str(partition_filters)
            ]
            if len(expression_candidates) == 0:
                logger.warning(
                    f"Partition filters are NOT compatible with partition columns, got: "
                    f"partition_filters={partition_filters} and partition_columns={partition_columns}"
                )
        filters = self.dataset_config.filters
        if partition_columns and filters is not None:
            expression_candidates = [x for x in partition_columns if x in str(filters)]
            if len(expression_candidates) > 0:
                logger.warning(
                    f"Partitionning columns {expression_candidates} are used as `filters` {filters}. ",
                    "You may want to use them in `partition_filters` instead",
                )

    def get_dataset(self) -> pq.ParquetDataset:
        if isinstance(self.dataset_config.filters, str):
            self.dataset_config.filters = pq.filters_to_expression(
                eval(self.dataset_config.filters)
            )

        if isinstance(self.dataset_config.partition_filters, str):
            self.dataset_config.partition_filters = pq.filters_to_expression(
                eval(self.dataset_config.partition_filters)
            )

        pq_ds = define_parquet_dataset(
            str(self.dataset_config.parquet_path), self.dataset_config.partition_filters
        )

        try:
            self._warn_filters_usage(pq_ds)
        except Exception as e:
            logger.info(f"getting exception during filters examination : {e}")

        return pq_ds

    def set_validation_params(
        self,
        world_size: int,
        default_max_tokens: int = 3000,
        default_batch_size: int = 40,
    ) -> None:
        if not (
            self.loading_config.batch_size is None
            and self.loading_config.max_tokens is None
        ):
            return

        total_batch_size = int(
            self.basic_stats.min_number_of_fragment
            * self.basic_stats.mean_fragment_length
        )
        batch_size = total_batch_size // world_size + int(
            total_batch_size % world_size != 0
        )

        # for small datasets we can set `batch_size`
        if (
            batch_size <= default_batch_size
            or self.basic_stats.mean_fragment_number_of_tokens is None
        ):
            self.loading_config.batch_size = min(batch_size, default_batch_size)
            self.loading_config.max_tokens = None
        else:
            # for bigger dataset, let's use `max_tokens`
            self.loading_config.batch_size = None
            total_tokens_number = int(
                self.basic_stats.min_number_of_fragment
                * self.basic_stats.mean_fragment_number_of_tokens
            )
            self.loading_config.max_tokens = min(
                max(total_tokens_number // world_size, 1), default_max_tokens
            )

    def build_dataload_pipeline(
        self, rank: int = 0, world_size: int = 1
    ) -> DataPipelineBuilder:
        if world_size > 1:
            assert self.loading_config.seed is not None, (
                "for distributed training with `world_size` > 1,  `seed` should be set !"
            )
        if self.is_validation:
            self.set_validation_params(world_size)

        # to propagate sharding_in_memory
        if not self.dataset_config.sharding_in_memory:
            sharding_in_memory = (
                self.loading_config.nb_epochs * self.proxy_number_of_fragments
                < 2 * world_size
            )
        else:
            sharding_in_memory = self.dataset_config.sharding_in_memory
        if self.loading_config.even_sharding:
            sharding_in_memory = True

        if sharding_in_memory:
            logger.info("Activating sharding_in_memory")

        self.random_state = np.random.RandomState(
            self._get_inner_seed(rank, sharding_in_memory)
        )
        pipeline = self.get_fragments_pipeline()

        if not sharding_in_memory:
            pipeline = pipeline.shard(
                shard_idx=rank,
                num_shards=world_size,
                allow_uneven=not self.loading_config.even_sharding,
            )

        pipeline = self.add_basic_fragment_loading_pipeline(pipeline)

        pipeline = self.create_on_the_fly_columns(pipeline)
        pipeline = self.filter_by_aligned_length(pipeline)

        # If we want to wrap before adding affixes
        if self.loading_config.wrap_before_affixing:
            pipeline = self.add_wrapping_to_max_length_pipeline(pipeline)

        # Filtering
        pipeline = self.add_quality_score_filters(pipeline)
        pipeline = self.add_min_sentence_number_in_doc_filter(
            pipeline,
            min_source_length=self.loading_config.min_length_of_sequences,
            min_target_length=self.loading_config.min_length_of_target_sequences,
        )
        pipeline = self.add_min_max_sentence_len_in_doc_filter(pipeline)

        # Affix
        pipeline = self._add_source_target_affixes_to_pipeline(pipeline)

        def cost_fn(table) -> float:
            cost = 0
            for name in [
                self.dataset_config.source_column,
                self.dataset_config.target_column,
            ]:
                if name is not None:
                    col = table[name]
                    if is_list_like(col):
                        cost += pa.compute.list_value_length(col).to_numpy().sum()
                    else:
                        # we should not be there, but let take batch_size as a proxy
                        cost += len(col)
            return cost

        pipeline = pipeline.dynamic_bucket(
            self._shuffling_tokens_size,
            cost_fn,
            min_num_examples=self.nb_parallel_fragments,
            max_num_examples=100,  # max number of small fragements
            drop_remainder=False,
        )
        pipeline = pipeline.map(concat_table, num_parallel_calls=1)

        # wrap documents after affixing
        if not self.loading_config.wrap_before_affixing:
            # Note that packing with proper attention masks and position codes requires
            # document indices that cover all sentences. Currently this can only come from affixing before wrapping.
            # Adding affixes after wrapping will require annexing these affixes to edge sentences which is not intuitive.
            if self.loading_config.shuffle:
                pipeline = pipeline.map(
                    partial(shuffle_table, random_state=self.random_state),
                    num_parallel_calls=1,
                )
            pipeline = self.add_wrapping_to_max_length_pipeline(pipeline)

        # batch with batch_size or max_tokens
        pipeline = self.add_inner_pipeline(pipeline)

        # Filter once again after wrapping and batching to remove batches with few number sentences
        pipeline = self.add_min_sentence_number_in_doc_filter(
            pipeline,
            min_source_length=self.loading_config.min_length_of_sequences_after_batching,
            min_target_length=self.loading_config.min_length_of_target_sequences_after_batching,
        )

        # Remove batch sizes with a size smaller than min_batch_size (default=1)
        pipeline = pipeline.filter(
            lambda table: bool(len(table) >= self.loading_config.min_batch_size)
        )

        if sharding_in_memory:
            pipeline = pipeline.shard(
                shard_idx=rank,
                num_shards=world_size,
                allow_uneven=not self.loading_config.even_sharding,
            )
        if self.loading_config.max_iteration_steps is not None:
            pipeline = pipeline.take(self.loading_config.max_iteration_steps)
        pipeline = self.add_format_conversion(pipeline)
        return pipeline

    def create_on_the_fly_columns(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        if self.dataset_config.source_sequences is not None:
            assert self.dataset_config.source_column is not None, (
                f"Expected a source_column - found {self.dataset_config.source_column}"
            )
            assert self.dataset_config.source_text_column is not None, (
                f"Expected a source_text_column - found {self.dataset_config.source_text_column}"
            )

            pipeline = pipeline.map(
                partial(
                    materialize_sequence,
                    column_sequence=self.dataset_config.source_sequences,
                    vector_name=self.dataset_config.source_column,
                    text_name=self.dataset_config.source_text_column,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            )
        if self.dataset_config.target_sequences is not None:
            assert self.dataset_config.target_column is not None, (
                f"Expected a target_column, found {self.dataset_config.target_column}"
            )
            assert self.dataset_config.target_text_column is not None, (
                f"Expected a target_text_columns, found {self.dataset_config.target_text_column}"
            )

            pipeline = pipeline.map(
                partial(
                    materialize_sequence,
                    column_sequence=self.dataset_config.target_sequences,
                    vector_name=self.dataset_config.target_column,
                    text_name=self.dataset_config.target_text_column,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            )

        columns_to_drop = list(
            set(self._get_sequences_columns()) - set(self.extra_required_columns)
        )
        if columns_to_drop:
            pipeline = pipeline.map(lambda table: table.drop(columns_to_drop))

        return pipeline

    def _add_source_target_affixes_to_pipeline(self, pipeline) -> DataPipelineBuilder:
        # prefixing/suffixing before wrapping/packing
        ps_vals = self._get_suffix_prefix_vector()
        pipeline = self.add_prefix_suffix_pipeline(
            pipeline,
            self.dataset_config.source_column,
            ps_vals["source_prefix_vector"],
            ps_vals["source_suffix_vector"],
        )
        pipeline = self.add_prefix_suffix_pipeline(
            pipeline,
            self.dataset_config.source_text_column,
            ps_vals["source_prefix_sentences"],
            ps_vals["source_suffix_sentences"],
        )

        pipeline = self.add_prefix_suffix_pipeline(
            pipeline,
            self.dataset_config.source_quality_column,
            (
                pa.array([None])
                if self.dataset_config.source_prefix_text
                else pa.array([])
            ),
            (
                pa.array([None])
                if self.dataset_config.source_suffix_text
                else pa.array([])
            ),
        )

        pipeline = self.add_prefix_suffix_pipeline(
            pipeline,
            self.dataset_config.target_column,
            ps_vals["target_prefix_vector"],
            ps_vals["target_suffix_vector"],
        )
        pipeline = self.add_prefix_suffix_pipeline(
            pipeline,
            self.dataset_config.target_text_column,
            ps_vals["target_prefix_sentences"],
            ps_vals["target_suffix_sentences"],
        )

        return pipeline

    def _num_parallel_call(self, x: float) -> int:
        return int(max(self.loading_config.num_parallel_calls * x, 1))

    def _nb_prefetch(self, x: float) -> int:
        return int(max(self.loading_config.nb_prefetch * x, 0))

    def config_post_init(self) -> None:
        if getattr(self.loading_config, "len_to_wrap_long_seq", None):
            if (
                self.dataset_config.target_column
                or self.dataset_config.target_text_column
            ):
                raise ValueError(
                    "Using `len_to_wrap_long_seq` is not supported for suppervised training"
                )

        if self.loading_config.even_sharding:
            assert self.loading_config.seed is not None, (
                "`even_sharding` sharding requires to seed to be set"
            )

        if self.loading_config.max_tokens == 0:
            self.loading_config.max_tokens = None
        # setting max_tokens=0 turns off this option (argparser won't accept None directly)

        if (self.loading_config.batch_size is None) == (
            self.loading_config.max_tokens is None
        ) and (not self.is_validation or self.loading_config.max_tokens is not None):
            raise ValueError(
                f"Need to provide either `batch_size` or `max_tokens` - \
                Received batch_size={self.loading_config.batch_size} \
                and max_tokens={self.loading_config.max_tokens}"
            )

        if self.loading_config.max_tokens and not self.dataset_config.source_column:
            raise ValueError(
                "Cannot batch based on `max_tokens` when `source_column` is not specified, "
                "please use `batch_size` instead."
            )

        self.dataset_config.split_to_row_groups = (
            self.dataset_config.split_to_row_groups
            if self.dataset_config.split_to_row_groups is not None
            else True
        )
        self.extra_required_columns = self.dataset_config.columns or []
        self.dataset_config.override_attr("columns", self._get_minimal_columns())
        logger.info(f"Following columns will be loaded: {self.dataset_config.columns}")

        self.basic_stats = self.compute_stats()

        self._shuffling_tokens_size = self._get_shuffling_tokens_size(self.basic_stats)
        logger.info(
            f"Bucketing will require at least: {self._shuffling_tokens_size} of tokens (source + target)"
        )
        logger.info(f"Dataset stats: {asdict(self.basic_stats)}")

        self.proxy_number_of_fragments = self.basic_stats.min_number_of_fragment
        if self.dataset_config.nb_parallel_fragments is None:
            self.dataset_config.nb_parallel_fragments = (
                self._find_nb_parallel_fragments(self.basic_stats)
            )

        logger.info(f"Dataset Config: {self.dataset_config}")
        logger.info(f"Using Loading Config: {self.loading_config}")

    def _get_shuffling_tokens_size(self, basic_stats) -> int:
        """
        `_shuffling_tokens_size` is used in dynamic bucketing to determine how many small parquet tables
        (which are loaded raw parquet fragments that were potentially filtered on-the-fly) will be merged together :
        we'll get a such number of consecutive parquet tables so that their total number of tokens (sentences)
        will be greater than `_shuffling_tokens_size`.
        It's called "shuffling" because all merged documents (from different tables) will be permuated together (if `shuffle=True`)
        before being returned as final small batches (of required shape or volume).

        The formula behind `_shuffling_tokens_size` is the following:
        - If we use `max_tokens` in config, we want to have a least _shuffling_tokens_size = 4 * max_tokens,
            so that at least 4 full batch will be formed next. It's good for shuffling and to avoid having "remainders" too often.
        - For wrapping/packing case, we use a proxy for `max_tokens` as `batch_size` * `len_to_wrap_long_seq`
        - If not, some average fragment characteristic `mean_fragment_number_of_tokens`, multiplied by 1.5 to get on average >=2 tables
        - Finally, if no, other info is available, we use 10_000 as arbitrary proxy (good typical value for many of our datasets).

        """
        if self.loading_config.max_tokens is not None:
            return 4 * self.loading_config.max_tokens
        if (
            self.loading_config.batch_size is not None
            and self.loading_config.len_to_wrap_long_seq is not None
        ):
            return (
                4
                * self.loading_config.len_to_wrap_long_seq
                * self.loading_config.batch_size
            )

        if basic_stats.mean_fragment_number_of_tokens is not None:
            return int(
                1.5 * basic_stats.mean_fragment_number_of_tokens
            )  # to get few fragments grouped together

        return 10_000  # default number that should not take a lot of RAM

    def _find_nb_parallel_fragments(
        self, basic_stats: GlobalPQStats, max_fragments=20, min_fragments=2
    ) -> int:
        """
        Experimental!
        Allows to determine nb of parallel fragments to load base on simple rules and dataset row group stats.
        In particular, if `nb_parallel_fragments` will increase with increasing batch_size of max_tokens.
        """
        if basic_stats.min_number_of_fragment < 3:
            return basic_stats.min_number_of_fragment

        if basic_stats.mean_fragment_number_of_tokens is None:
            logger.warning(
                f"Cannot get `mean_fragment_number_of_tokens` from dataset {self.dataset_config}, `nb_parallel_fragement` detection can be wrong",
            )

        mean_fragment_number_of_tokens = (
            basic_stats.mean_fragment_number_of_tokens or 5000
        )  # typical, but arbitrary value
        if (
            self.loading_config.batch_size is None
            and self.loading_config.max_tokens is None
        ):
            # it can happen for evaluation
            nb_frags = 1.0
        elif self.loading_config.batch_size is not None:
            if self.loading_config.len_to_wrap_long_seq is not None:
                max_tokens = (
                    self.loading_config.len_to_wrap_long_seq
                    * self.loading_config.batch_size
                )
                nb_frags = 3 * max_tokens / mean_fragment_number_of_tokens
            else:
                nb_frags = (
                    5
                    * self.loading_config.batch_size
                    / basic_stats.mean_fragment_length
                )
        elif self.loading_config.max_tokens is not None:
            nb_frags = (
                3 * self.loading_config.max_tokens / mean_fragment_number_of_tokens
            )

        return max(min(max_fragments, round(nb_frags)), min_fragments)

    @lru_cache
    def _get_sequences_columns(self):
        candidate_columns = []
        for col in (self.dataset_config.source_sequences or []) + (
            self.dataset_config.target_sequences or []
        ):
            candidate_columns.append(col.text_column)
            candidate_columns.append(col.sonar_column)
        return [x for x in candidate_columns if x is not None]

    def _get_minimal_columns(self):
        # restrict on used collumns
        candidate_columns = [
            self.dataset_config.source_column,
            self.dataset_config.source_text_column,
            self.dataset_config.source_quality_column,
            self.dataset_config.target_column,
            self.dataset_config.target_text_column,
            "split",
        ] + self._get_sequences_columns()

        minimal_columns: List[str] = [
            x
            for x in candidate_columns
            if x is not None and x in self.full_schema.names
        ]

        if self.dataset_config.columns is None:
            columns = sorted(set(minimal_columns))
        else:
            columns = sorted(set(minimal_columns + list(self.dataset_config.columns)))
        if not set(columns).issubset(set(self.full_schema.names)):
            raise ValueError(
                f"columns {sorted(set(columns) - set(self.full_schema.names))} are not found in the dataset schema"
            )

        return columns

    def _get_suffix_prefix_vector(self):
        nested_result = prepare_suffix_prefix_embeddings(
            self.dataset_config.source_prefix_text,
            self.dataset_config.source_suffix_text,
            self.dataset_config.target_prefix_text,
            self.dataset_config.target_suffix_text,
        )

        names = (
            ("source_prefix_vector", "source_prefix_sentences"),
            ("source_suffix_vector", "source_suffix_sentences"),
            ("target_prefix_vector", "target_prefix_sentences"),
            ("target_suffix_vector", "target_suffix_sentences"),
        )

        return {n: v for nn, val in zip(names, nested_result) for n, v in zip(nn, val)}

    def get_fragments_pipeline(self):
        split_to_row_groups = self.dataset_config.split_to_row_groups
        assert isinstance(split_to_row_groups, bool)

        # one can use `list_parquet_fragments` for a full fragments scan
        fragments_pipeline_builder = stream_parquet_fragments(
            parquet_ds=self.dataset,
            nb_epochs=self.loading_config.nb_epochs,
            split_to_row_groups=split_to_row_groups,
            shuffle=self.loading_config.shuffle,
            seed=self.loading_config.seed,
            limit_options=self.dataset_config.limit,
            shuffling_window=20 * self.nb_parallel_fragments,
        )

        return fragments_pipeline_builder

    def compute_stats(self, max_fragments=100) -> GlobalPQStats:
        if self.dataset_config.source_sequences:
            source_column = None
        else:
            source_column = self.dataset_config.source_column

        split_to_row_groups = self.dataset_config.split_to_row_groups

        columns = [source_column] if source_column else None

        if (
            self.dataset_config.limit is not None
            and self.dataset_config.limit.nb_fragments is not None
        ):
            # TODO: take into account other limit options to get better estimates
            max_fragments = min(self.dataset_config.limit.nb_fragments, max_fragments)

        self._stats_df = get_row_group_level_metadata(
            self.dataset, columns=columns, max_fragments=max_fragments
        )
        dim = 1
        if source_column:
            self._stats_df["num_tokens"] = self._stats_df[source_column].apply(
                lambda x: x["num_values"]
            )

            type_source = self.full_schema.field(source_column).type
            try:
                dim = type_source.value_type.list_size
                if not dim or dim < 0:
                    dim = 1  # not a fixed vector size
            except AttributeError:
                logger.warning(f"source column {source_column} is not of list type")
                if self.dataset_config.nb_parallel_fragments is None:
                    logger.warning("you may need to provide `nb_parallel_fragments`")
                dim = 1

        if split_to_row_groups:
            global_stats_df = self._stats_df
        elif "num_tokens" in self._stats_df:
            global_stats_df = self._stats_df.groupby("parquet_file_path").agg(
                {"num_rows": "sum", "num_tokens": "sum"}
            )
        else:
            global_stats_df = self._stats_df.groupby("parquet_file_path").agg(
                {"num_rows": "sum"}
            )

        mean_len_frag = global_stats_df["num_rows"].mean()

        if "num_tokens" in global_stats_df:
            mean_num_tokens_frag = self._stats_df["num_tokens"].mean() / dim
        else:
            mean_num_tokens_frag = None

        return GlobalPQStats(
            len(global_stats_df),
            mean_len_frag,
            mean_fragment_number_of_tokens=mean_num_tokens_frag,
        )

    def add_inner_pipeline(self, pipeline: DataPipelineBuilder) -> DataPipelineBuilder:
        loading_config = self.loading_config

        columns_to_bucket = [
            self.dataset_config.source_column,
            self.dataset_config.target_column,
        ]
        columns_to_bucket = [x for x in columns_to_bucket if x is not None]

        def inner_iterator(table: pa.Table) -> DataPipeline:
            return build_batching_loop_over_one_table(
                table=table,
                order_by_length=self.loading_config.order_by_length,
                length_column=columns_to_bucket,
                batch_size=loading_config.batch_size,
                max_tokens=loading_config.max_tokens,
                shuffle=loading_config.shuffle,
                seed=self.random_state.randint(0, 2**32),
                num_parallel_calls=self._num_parallel_call(3),
            )

        return pipeline.yield_from(inner_iterator)

    def _get_inner_seed(self, rank: int, sharding_in_memory: bool) -> Optional[int]:
        if self.loading_config.seed is not None:
            if not sharding_in_memory:
                return int(self.loading_config.seed) + rank * 100_000
            else:
                # for `sharding_in_memory`, we want the same shuffling
                # to guarantee the consistent sharding across ranks
                return int(self.loading_config.seed)
        else:
            return None

    def add_prefix_suffix_pipeline(
        self,
        pipeline: DataPipelineBuilder,
        column: Optional[str],
        prefix,
        suffix,
    ) -> DataPipelineBuilder:
        if (suffix is None and prefix is None) or column is None:
            return pipeline
        pipeline = pipeline.map(
            partial(
                prefix_and_suffix_one_list_column,
                column=column,
                prefix_array=prefix,
                suffix_array=suffix,
            ),
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        )
        return pipeline

    def add_basic_fragment_loading_pipeline(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        def load_fn(safe_frag):
            try:
                return safe_frag.load(columns=self.dataset_config.columns)
            except Exception as e:
                logger.error(
                    f"Error {e} occured while loading fragment {safe_frag} \n, skipping it"
                )
                return None

        pipeline = pipeline.map(
            load_fn,
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        )

        pipeline = pipeline.filter(lambda table: bool(table is not None))

        # we reapply the partition filters just in case of misusage
        # but it should not change the performance
        partition_filters = self.dataset_config.partition_filters
        filters = self.dataset_config.filters
        if partition_filters is not None and filters is not None:
            full_filter = pa.compute.if_else(filters, partition_filters, False)
        else:
            full_filter = partition_filters if filters is None else filters

        pipeline = pipeline.map(
            partial(
                apply_filter,
                filters=full_filter,
                drop_null=self.loading_config.drop_null,
            )
        )

        pipeline = pipeline.filter(lambda table: bool(len(table) > 0))
        pipeline = pipeline.prefetch(self._nb_prefetch(self.nb_parallel_fragments))

        return pipeline

    def filter_by_aligned_length(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        source_columns: List[str] = [
            x
            for x in (
                self.dataset_config.source_column,
                self.dataset_config.source_text_column,
                self.dataset_config.source_quality_column,
            )
            if x is not None
        ]

        # filter out sample where number of sentences and number of sonar embeddings are not equal
        # which should never happen normally

        pipeline = pipeline.map(
            partial(
                filter_table_with_different_lengths,
                columns=source_columns,
            ),
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        )
        pipeline = pipeline.filter(lambda table: bool(len(table) > 0))

        target_columns: List[str] = [
            x
            for x in (
                self.dataset_config.target_column,
                self.dataset_config.target_text_column,
            )
            if x is not None
        ]

        pipeline = pipeline.map(
            partial(
                filter_table_with_different_lengths,
                columns=target_columns,
            ),
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        )
        pipeline = pipeline.filter(lambda table: bool(len(table) > 0))

        return pipeline

    def add_wrapping_to_max_length_pipeline(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        len_to_wrap_long_seq = getattr(
            self.loading_config, "len_to_wrap_long_seq", None
        )
        if len_to_wrap_long_seq is None:
            return pipeline

        columns_to_wrap: List[str] = [
            x
            for x in (
                self.dataset_config.source_column,
                self.dataset_config.source_text_column,
                self.dataset_config.source_quality_column,
            )
            if x is not None
        ]

        if self.loading_config.packing:
            method = return_none_on_failure(explode_table_with_fixed_length)
            logger.info(
                f"Wrapping to len_to_wrap_long_seq={len_to_wrap_long_seq} with fixed length (packing)"
            )
        else:
            method = return_none_on_failure(explode_table_with_max_length)
            logger.info(
                f"Wrapping to len_to_wrap_long_seq={len_to_wrap_long_seq} with max length (without packing)"
            )

        pipeline = pipeline.map(
            partial(
                method,
                columns=columns_to_wrap,
                max_seq_len=len_to_wrap_long_seq,
            ),
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        )
        return pipeline.filter(lambda table: table is not None)

    def add_min_max_sentence_len_in_doc_filter(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        if (
            self.loading_config.max_sentence_len_in_doc
            or self.loading_config.min_sentence_len_in_doc
        ):
            assert self.dataset_config.source_text_column is not None, (
                f"Expexted a source_text_columns, found {self.dataset_config.source_text_column}"
            )

            pipeline = pipeline.map(
                partial(
                    filter_long_short_sentence_document,
                    column=self.dataset_config.source_text_column,
                    max_sentence_len=self.loading_config.max_sentence_len_in_doc,
                    min_sentence_len=self.loading_config.min_sentence_len_in_doc,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            ).filter(lambda table: bool(len(table) > 0))

        if self.dataset_config.target_column is not None and (
            self.loading_config.max_sentence_len_in_target_doc
            or self.loading_config.min_sentence_len_in_target_doc
        ):
            pipeline = pipeline.map(
                partial(
                    filter_long_short_sentence_document,
                    column=self.dataset_config.target_column,
                    max_sentence_len=self.loading_config.max_sentence_len_in_target_doc,
                    min_sentence_len=self.loading_config.min_sentence_len_in_target_doc,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            ).filter(lambda table: bool(len(table) > 0))

        return pipeline

    def add_min_sentence_number_in_doc_filter(
        self,
        pipeline: DataPipelineBuilder,
        min_source_length: Optional[int] = None,
        min_target_length: Optional[int] = None,
    ) -> DataPipelineBuilder:
        """
        If `min_source_length` is not None: filter the source to remove sequences
            with less than `min_source_length` sentences
        If `min_target_length` is not None and data comes with a target column:
            filter the target to remove sequences with less than `min_target_length` sentences

        """

        def _min_length_filter(table, column, length):
            filter_ = pc.greater_equal(pc.list_value_length(table[column]), length)

            if pc.all(filter_).as_py():
                return table
            return table.filter(filter_)

        if (
            self.dataset_config.source_column is not None
            and min_source_length is not None
        ):
            pipeline = pipeline.map(
                partial(
                    _min_length_filter,
                    column=self.dataset_config.source_column,
                    length=min_source_length,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            ).filter(lambda table: bool(len(table) > 0))

        if (
            self.dataset_config.target_column is not None
            and min_target_length is not None
        ):
            pipeline = pipeline.map(
                partial(
                    _min_length_filter,
                    column=self.dataset_config.target_column,
                    length=min_target_length,
                ),
                num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
            ).filter(lambda table: bool(len(table) > 0))

        return pipeline

    def add_quality_score_filters(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        source_quality_range = self.dataset_config.source_quality_range
        if source_quality_range is None:
            return pipeline

        assert self.dataset_config.source_quality_column is not None, (
            f"Expected a source_quality_columns, found {self.dataset_config.source_quality_column}"
        )

        pipeline = pipeline.map(
            partial(
                filter_document_by_quality,
                column=self.dataset_config.source_quality_column,
                min_score=source_quality_range[0],
                max_score=source_quality_range[1],
            ),
            num_parallel_calls=self._num_parallel_call(self.nb_parallel_fragments),
        ).filter(lambda table: bool(len(table) > 0))
        return pipeline

    def add_format_conversion(
        self, pipeline: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        if self.loading_config.output_format == ParquetBatchFormat.pandas:
            pipeline = pipeline.map(lambda table: table.to_pandas())
        elif self.loading_config.output_format == ParquetBatchFormat.torch:
            pipeline = pipeline.map(lambda wt: pyarrow_table_to_torch_dict(wt))
        return pipeline

    def get_python_iterator(
        self, rank: int = 0, world_size: int = 1
    ) -> Generator[BatchOutputType, None, None]:  # type: ignore
        yield from iter(
            self.build_dataload_pipeline(
                rank=rank,
                world_size=world_size,
            )
            .prefetch(self._nb_prefetch(5))
            .and_return(max_num_warnings=4)
        )


def parquet_iterator(
    dataset_config: ParquetDatasetConfig,
    loading_config: DataLoadingConfig,
    rank: int,
    world_size: int,
) -> Generator[BatchOutputType, None, None]:  # type: ignore
    spdd = SingleParquetDatasetDataloader(dataset_config, loading_config)
    yield from spdd.get_python_iterator(rank, world_size)


def build_parquet_iterator_pipeline(
    dataset_config: ParquetDatasetConfig,
    loading_config: DataLoadingConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataPipelineBuilder:
    return SingleParquetDatasetDataloader(
        dataset_config, loading_config
    ).build_dataload_pipeline(rank=rank, world_size=world_size)


def ds_name(conf: ParquetDatasetConfig) -> str:
    if conf.name is not None:
        return conf.name
    return str(conf.parquet_path)


def circular_shift_left(lst: List[Any], k: int) -> List[Any]:
    if len(lst) <= 1:
        return lst

    k = k % len(lst)  # To handle shifts larger than the list length
    return lst[k:] + lst[:k]


def build_weighted_pipeline_with_renaming(
    dataset_configs: Sequence[ParquetDatasetConfig],
    loading_config: DataLoadingConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataPipeline:
    assert loading_config.multiple_dataset_chaining in [
        "sample",
        "concat",
        "round_robin",
    ]

    # adjusting the number parallel calls and prefetch according to total number of datasets
    dataset_configs = list(dataset_configs)
    loading_config.num_parallel_calls = loading_config.num_parallel_calls / len(
        dataset_configs
    )
    loading_config.nb_prefetch = loading_config.nb_prefetch // len(dataset_configs)

    name_mappers = get_renaming_mappers(dataset_configs)
    pipelines: List[DataPipelineBuilder] = []

    def process_one_pipeline(cc, mapper):
        return build_parquet_iterator_pipeline(
            dataset_config=cc,
            loading_config=loading_config,
            rank=rank,
            world_size=world_size,
        ).map(
            partial(renaming, mapper=mapper, name=ds_name(cc)),
            num_parallel_calls=1,
        )

    # creating all datasets pipeline in parallel
    pipelines = [
        process_one_pipeline(cc, mapper)
        for cc, mapper in zip(dataset_configs, name_mappers)
    ]

    if len(pipelines) == 1:
        return (
            pipelines[0]
            .prefetch(int(max(loading_config.nb_prefetch, 1)))
            .and_return(max_num_warnings=4)
        )
    if loading_config.seed is not None:
        seed = loading_config.seed + (0 if loading_config.even_sharding else rank)
    else:
        seed = None

    pipelines_with_return = [pp.and_return(max_num_warnings=4) for pp in pipelines]

    if loading_config.multiple_dataset_chaining == "concat":
        # TODO : check that all weights = 1
        weighted_pipeline = DataPipeline.concat(
            circular_shift_left(pipelines_with_return, k=rank),
        )
    elif loading_config.multiple_dataset_chaining == "round_robin":
        weighted_pipeline = DataPipeline.round_robin(
            circular_shift_left(pipelines_with_return, k=rank), allow_repeats=False
        )
    else:
        weighted_pipeline = DataPipeline.sample(
            pipelines_with_return,
            [getattr(cc, "weight", 1.0) for cc in dataset_configs],
            seed=seed,
        )

    return weighted_pipeline.prefetch(
        int(
            max(loading_config.nb_prefetch * len(dataset_configs) ** 2, 1)
        )  # try to prefetch at least one element from each dataset
    ).and_return(max_num_warnings=4)
