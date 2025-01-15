# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#


import logging
from dataclasses import dataclass
from functools import lru_cache, reduce, wraps
from pickle import dumps, loads
from typing import Any, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from fairseq2.data.data_pipeline import (
    DataPipeline,
    DataPipelineBuilder,
    read_iterator,
    read_sequence,
)
from fairseq2.data.parquet.tools import (
    NestedDict,
    NestedDictValue,
    add_partitioning_values,
    compute_rows_length,
    get_dataset_fragments,
    split_fragment_in_row_groups,
)
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pyarrow.dataset import get_partition_keys
from retrying import retry
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextBatchEmbedder,
    SonarTextEmbedderConfig,
)
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo
from stopes.utils.arrow_utils import (
    hstack_pyarray_list,
    is_list_like,
    pyarrow_column_to_array,
    simple_array_to_nested,
)
from tqdm.auto import tqdm

from lcm.datasets.configs import (
    ColumnsNames,
    ParquetDatasetLimitOptions,
    SonarTextColumn,
)
from lcm.utils.common import batched

try:
    from numba import njit
except ModuleNotFoundError:
    print("Numba is not installed. Fall-back to the non-recompiled version")

    def empty_jit(f):
        @wraps(f)
        def _f(*args, **kwargs):
            return f(*args, **kwargs)

        return _f

    njit = empty_jit


loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, OSError),
    stop_max_attempt_number=1,
    wait_exponential_multiplier=2,
    wait_exponential_max=20,
)


logger = logging.getLogger(__name__)


def prefix_and_suffix_one_list_column(
    table: pa.Table, column: str, prefix_array: pa.Array, suffix_array: pa.Array
):
    prefix_extended = pa.chunked_array(
        [pa.ListArray.from_arrays([0, len(prefix_array)], prefix_array)] * len(table)
    )
    suffix_extended = pa.chunked_array(
        [pa.ListArray.from_arrays([0, len(suffix_array)], suffix_array)] * len(table)
    )
    target_dtype = table[column].type
    if prefix_extended.type != target_dtype:
        prefix_extended = prefix_extended.cast(target_dtype)
    if suffix_extended.type != target_dtype:
        suffix_extended = suffix_extended.cast(target_dtype)

    new_array = hstack_pyarray_list(prefix_extended, table[column], suffix_extended)
    return table.drop([column]).append_column(column, new_array)


def define_parquet_dataset(parquet_path: str, partition_filters) -> pq.ParquetDataset:
    return pq.ParquetDataset(
        parquet_path,
        filters=partition_filters,
    )


@lru_cache()
def default_sonar_pipeline() -> SonarTextBatchEmbedder:
    local_sonar_config = SonarTextEmbedderConfig(
        column_config=[
            LangColumnConfig("input_text", lang_value="eng_Latn"),
        ],
        batch_size=10,
        device="cpu",
    )
    return SonarTextBatchEmbedder(local_sonar_config)


@lru_cache(2000)
def _get_embed_sentences(text: Optional[str]) -> pa.Array:
    sentences_splitter = get_split_algo("eng_Latn", "default")
    lstbe = default_sonar_pipeline()
    sentences = pa.array(sentences_splitter(text) if text else [""])
    input_table = pa.Table.from_pydict({"input_text": sentences})
    vectors = pyarrow_column_to_array(lstbe(input_table)["input_text_sonar_emb"])
    if not text:
        # empty output of the right type
        vectors = vectors.slice(0, 0)
        sentences = sentences.slice(0, 0)
    return vectors, sentences


def prepare_suffix_prefix_embeddings(*args):
    if all(xx is None for xx in args):  # to avoid loading SonarModel
        return [(None, None) for _ in args]

    return [_get_embed_sentences(xx) for xx in args]


def from_pyarrow_to_torch_tensor(
    arr: Union[pa.Array, pa.ChunkedArray], strict: bool = False
) -> NestedDictValue:
    """
    struct_array = pa.Array.from_pandas([{"x": 4, "y": "RR"}] * 10)
    nest_array = pa.Array.from_pandas([[{'a': 1}, {'a': 2}]])
    """
    # for future ideas https://arrow.apache.org/docs/python/generated/pyarrow.Tensor.html
    # for sparse matrix support https://github.com/apache/arrow/blob/main/python/pyarrow/tests/test_sparse_tensor.py

    if arr.null_count != 0:
        raise ValueError("to torch conversion does not support null values")

    arr = pyarrow_column_to_array(arr)

    arr_type = arr.type
    if pa.types.is_primitive(arr_type):
        try:
            return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
        except Exception:
            pass

    try:
        return torch.from_numpy(arr.to_numpy(zero_copy_only=True))
    except pa.ArrowInvalid:
        pass

    if pa.types.is_dictionary(arr_type):
        return from_pyarrow_to_torch_tensor(arr.dictionary_decode())

    if pa.types.is_string(arr_type):
        return arr.to_pandas().tolist()

    if pa.types.is_list(arr_type) or pa.types.is_large_list(arr_type):
        if pa.types.is_primitive(arr_type.value_type):
            return arr.to_pandas().map(torch.from_numpy).tolist()

        if pa.types.is_fixed_size_list(arr_type.value_type) and pa.types.is_primitive(
            arr_type.value_type.value_type
        ):
            return (
                arr.to_pandas()
                .map(
                    lambda x: torch.from_numpy(
                        np.vstack(x) if len(x) > 0 else np.array([], dtype=np.float32)
                    )
                )
                .tolist()
            )

    if pa.types.is_fixed_size_list(arr_type):
        if pa.types.is_primitive(arr_type.value_type):
            return torch.from_numpy(np.reshape(arr.values, (-1, arr_type.list_size)))

    if pa.types.is_struct(arr_type):
        return {
            arr_type.field(i).name: from_pyarrow_to_torch_tensor(arr.field(i))
            for i in range(arr_type.num_fields)
        }

    if pa.types.is_nested(arr_type):
        # TODO: deal with arr = [[{'a': 1}, {'a': 2}]]
        pass

    if strict:
        raise NotImplementedError(f"{arr_type} cannot be converted to torch.Tensor")
    else:
        return arr  # keeping as in the orignal pyarrow form


def pyarrow_table_to_torch_dict(tt: pa.Table, strict: bool = False) -> NestedDict:
    out = {}
    for col in tt.column_names:
        try:
            out[col] = from_pyarrow_to_torch_tensor(tt[col], strict)
        except ValueError as e:
            logger.info(
                f"Column {col} of type {tt[col].type} was not converted to torch as expected",
                str(e),
            )
            out[col] = tt[col]
    return out


def add_fragments_trace(table: pa.Table, fragment: pa.dataset.Fragment) -> pa.Table:
    table = table.append_column(
        "__row_groups_ids",
        len(table)
        * [np.array([int(rg.id) for rg in fragment.row_groups], dtype=np.int32)],
    )
    table = table.append_column(
        "__index_in_fragement", pa.array(np.arange(len(table), dtype=np.int32))
    )
    return table


def shuffle_table(table: pa.Table, random_state: np.random.RandomState) -> pa.Table:
    permutation = pa.array(random_state.permutation(len(table)))
    return table.take(permutation)


class SafeFragment:
    """
    Experimental :
    Simple wrapper around `ParquetFileFragment` that allows to reinit the state of filesystem
    if aws session token has expired.
    """

    fragment: pa.dataset.ParquetFileFragment

    def __init__(self, fragment: pa.dataset.ParquetFileFragment):
        self.fragment = fragment

    def __repr__(self) -> str:
        out = ""
        out += "SafeFragment \n"
        out += "path = " + self.fragment.path + "\n"
        out += f"row_groups = {[int(rg.id) for rg in self.fragment.row_groups]} \n"
        out += f"physical_schema = \n {self.fragment.physical_schema} \n"
        return out

    @loading_retry
    def load(self, columns: Optional[List[str]] = None) -> pa.Table:
        if columns is not None:
            fragment_columns = [
                col for col in columns if col in self.fragment.physical_schema.names
            ]
        else:
            fragment_columns = self.fragment.physical_schema.names
        # adding technical columns for tracking
        fragment_columns = list(fragment_columns) + [
            "__batch_index",
            "__fragment_index",
            "__filename",
        ]
        try:
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=False
            )

        except OSError as e:
            logger.info(
                "could not load fragment, reinit the fragment state. Error: ", str(e)
            )
            self.fragment = loads(dumps(self.fragment))
            fragment_table = self.fragment.to_table(
                columns=fragment_columns, use_threads=False
            )

        fragment_table = add_partitioning_values(fragment_table, self.fragment, columns)
        fragment_table = add_fragments_trace(fragment_table, self.fragment)
        return fragment_table


def _parquet_fragments_to_pipeline_builder(
    file_ds_fragments: List[pa.dataset.Fragment],
    nb_epochs: int = 1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataPipelineBuilder:
    if shuffle:
        if seed is None:
            seed = int(torch.randint(0, 2**31, ()).item())

        rsg = np.random.RandomState(seed)
        ds_fragments_ = np.asarray(file_ds_fragments, dtype="O")
        ds_fragments = np.concatenate(
            [rsg.permutation(ds_fragments_) for _ in range(nb_epochs)]
        ).tolist()
    else:
        ds_fragments = file_ds_fragments * nb_epochs

    pipeline_builder = read_sequence(ds_fragments)
    pipeline_builder = pipeline_builder.map(SafeFragment)
    return pipeline_builder


def list_parquet_fragments(
    parquet_ds: pq.ParquetDataset,
    nb_epochs: int = 1,
    split_to_row_groups: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = None,
    limit_options: Optional[ParquetDatasetLimitOptions] = None,
    nb_jobs: int = 10,
) -> DataPipelineBuilder:
    if limit_options is None:
        limit_options = ParquetDatasetLimitOptions()

    file_ds_fragments = get_dataset_fragments(parquet_ds, parquet_ds._filter_expression)
    proxy_ds_path = "/".join(parquet_ds.files[0].split("=")[0].split("/")[:-1])

    logger.info(f"{proxy_ds_path} : full number of files {len(file_ds_fragments)}")
    if limit_options.fraction_of_files is not None:
        file_ds_fragments = file_ds_fragments[
            : max(
                int(round(limit_options.fraction_of_files * len(file_ds_fragments))), 1
            )
        ]
        logger.info(
            f"{proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of fraction_of_files={limit_options.fraction_of_files}"
        )
    if limit_options.nb_files is not None and limit_options.nb_files < len(
        file_ds_fragments
    ):
        file_ds_fragments = file_ds_fragments[: limit_options.nb_files]
        logger.info(
            f"{proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of nb_files={limit_options.nb_files}"
        )

    output_fragments = []
    total_nb_rows = 0
    if split_to_row_groups:
        logger.info(f"{proxy_ds_path} : starting split in row groups")

        with Parallel(backend="threading", n_jobs=nb_jobs) as parallel:
            total_nb_fragments = 0
            early_stop = False

            for batch_of_files in batched(file_ds_fragments, 20 * nb_jobs):
                row_groups = parallel(
                    delayed(split_fragment_in_row_groups)(ff) for ff in batch_of_files
                )
                new_file_fragments = [x for y in row_groups for x in y]
                if limit_options.nb_rows is not None:
                    new_file_fragments_stats = parallel(
                        delayed(lambda frag: frag.row_groups[0].num_rows)(ff)
                        for ff in new_file_fragments
                    )
                else:
                    new_file_fragments_stats = [0] * len(new_file_fragments)

                for nb_row, frag in zip(new_file_fragments_stats, new_file_fragments):
                    output_fragments.append(frag)
                    total_nb_rows += nb_row
                    total_nb_fragments += 1
                    if (
                        limit_options.nb_fragments is not None
                        and total_nb_fragments >= limit_options.nb_fragments
                    ):
                        early_stop = True
                        if limit_options.nb_rows is not None:
                            logger.info(
                                f"{proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached with around {total_nb_rows} rows"
                            )
                        else:
                            logger.info(
                                f"{proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached"
                            )
                        break
                    if (
                        limit_options.nb_rows is not None
                        and total_nb_rows >= limit_options.nb_rows
                    ):
                        early_stop = True
                        logger.info(
                            f"{proxy_ds_path} : nb_rows limit {limit_options.nb_rows} was reached with around {total_nb_fragments} fragments"
                        )
                        break
                if early_stop:
                    break
    else:
        for frag in file_ds_fragments[: limit_options.nb_fragments]:
            output_fragments.append(frag)
            if limit_options.nb_rows is not None:
                total_nb_rows += frag.count_rows()
                if total_nb_rows >= limit_options.nb_rows:
                    break

    logger.info(f"{proxy_ds_path} : finding fragments {len(output_fragments)}")

    return _parquet_fragments_to_pipeline_builder(
        output_fragments,
        nb_epochs=nb_epochs,
        shuffle=shuffle,
        seed=seed,
    )


def compute_length_splits(
    length_col: NDArray[np.int32],
    max_tokens: int,
    order_by_length: bool = True,
    drop_long_sample: bool = True,
) -> List[NDArray[np.int32]]:
    """split sequence of length_col in the chunks such that total length is ~ max_tokens
        countint the padding to max length of elements in a chunk

    Args:
        length_col (np.ndarray):
        max_tokens (int):
        order_by_length (bool):
        drop_long_sample (bool):

    Returns:
        List[np.ndarray]: splits that contain indices over the original length_col
    """
    argsort_ind = (
        np.argsort(length_col)
        if order_by_length
        else np.arange(len(length_col), dtype=np.int32)
    )

    sorted_length_col = length_col[argsort_ind]

    small_elements_masks = sorted_length_col <= max_tokens
    big_elements_inds = argsort_ind[~small_elements_masks]

    argsort_ind = argsort_ind[small_elements_masks]
    sorted_length_col = sorted_length_col[small_elements_masks]

    size = len(sorted_length_col)
    splits = []
    begin, end = 0, 0
    while end < size:
        current_max_len = sorted_length_col[begin]
        begin = end
        while end < size:
            current_max_len = max(current_max_len, sorted_length_col[end])
            if current_max_len * (end + 1 - begin) > max_tokens:
                splits.append(argsort_ind[begin:end])
                break
            end += 1
    else:
        if begin < size:
            splits.append(argsort_ind[begin:])

    # adding big sample at the end one by one
    if not drop_long_sample and len(big_elements_inds):
        splits.extend(np.array_split(big_elements_inds, len(big_elements_inds)))

    return splits


def build_batching_loop_over_one_table(
    table: pa.Table,
    order_by_length: bool = False,
    length_column: List[Optional[str]] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    num_parallel_calls: int = 1,
) -> DataPipeline:
    if max_tokens is not None:
        assert length_column is not None, (
            "Need to provide a column to compute the number of tokens"
        )

    random_state = np.random.RandomState(seed)
    if length_column is not None and len(length_column) > 0:
        length_col = reduce(
            np.add, (compute_rows_length(table[lc]) for lc in length_column)
        )
    else:
        if shuffle:
            length_col = random_state.randint(0, 2**23, len(table))
        else:
            length_col = np.zeros(len(table), dtype=np.int32)

    if batch_size is not None:
        if order_by_length:
            sorting_ind = np.argsort(length_col, kind="stable")
        else:
            sorting_ind = np.arange(len(length_col), dtype=np.int32)

        order_tt = pa.Table.from_arrays([pa.array(sorting_ind)], ["order"])
        batches = [ind["order"] for ind in order_tt.to_batches(batch_size)]
    elif max_tokens is not None:
        batches = compute_length_splits(
            length_col, max_tokens, order_by_length=order_by_length
        )
    else:
        raise ValueError("unknown batching method")

    if shuffle:
        batches = [batches[i] for i in random_state.permutation(len(batches))]

    def _getter(ind):
        try:
            tt = table.take(ind)
            return tt
        except Exception as e:
            logger.warn(f"Unexpected error : \n {str(e)} \n {table} \n {ind}")
            return None

    return (
        read_sequence(batches)
        .map(_getter, num_parallel_calls=num_parallel_calls)
        .filter(lambda tt: bool(tt is not None))
        .and_return(max_num_warnings=4)
    )


def filter_long_short_sentence_document(
    batch: pa.Table,
    column: str,
    max_sentence_len: Optional[int],
    min_sentence_len: Optional[int],
) -> pa.Table:
    assert max_sentence_len is not None or min_sentence_len is not None
    if min_sentence_len is None:
        min_sentence_len = 0

    if max_sentence_len is None:
        max_sentence_len = 2**32

    tt = pl.from_arrow(batch.select([column]), rechunk=False)
    assert isinstance(tt, pl.DataFrame)
    filter_ = tt.with_columns(
        (
            pl.col(column).list.eval(pl.col("").str.len_bytes()).list.max()
            <= max_sentence_len
        )
        & (
            pl.col(column).list.eval(pl.col("").str.len_bytes()).list.min()
            <= max_sentence_len
        )
    )[column].to_arrow()

    if pc.all(filter_).as_py():
        return batch
    return batch.filter(filter_)


def filter_document_by_quality(
    batch: pa.Table,
    column: str,
    min_score=Optional[float],
    max_score=Optional[float],
) -> pa.Table:
    if min_score is None and max_score is None:
        return batch

    if min_score is None:
        min_score = -float(np.inf)
    if max_score is None:
        max_score = float(np.inf)

    tt = pl.from_arrow(batch.select([column]), rechunk=False)
    assert isinstance(tt, pl.DataFrame)
    filter_ = tt.with_columns(
        (pl.col(column).list.max() <= max_score)
        & (pl.col(column).list.min() >= min_score)
    )[column].to_arrow()
    if pc.all(filter_).as_py():
        return batch
    return batch.filter(filter_)


def renaming(inp: NestedDict, mapper: dict, name: str) -> NestedDict:
    renamed_name = ColumnsNames.dataset_name.value
    if isinstance(inp, dict):
        out_dict = {mapper.get(key, key): value for key, value in inp.items()}
        out_dict[renamed_name] = name
        res = out_dict
    elif isinstance(inp, pd.DataFrame):
        out_pd = inp.rename(mapper=mapper, axis=1)
        out_pd[renamed_name] = name
        res = out_pd
    elif isinstance(inp, pa.Table):
        out_pa: pa.Table = inp.rename_columns(
            [mapper.get(key, key) for key in inp.column_names],
        )
        out_pa = out_pa.append_column(renamed_name, pa.array([name] * len(out_pa)))
        res = out_pa
    return res


def materialize_sequence(
    table: pa.Table,
    column_sequence: List[SonarTextColumn],
    vector_name: str,
    text_name: str,
) -> pa.Table:
    """
    Given `table`, it materializes `column_sequence`.
    Different elements from `column_sequence` are concatenated sequentially.
    Constant text elements will be sentencized and sonarized.
    It also accepts columns with single text and embeddings values instead of list.

    It returns a new table with two new columns with sequences of sentences and corresponding sequences of their embeddings.
    """

    table_len = len(table)
    sentences_seq = []
    vectors_seq = []

    target_dtype = None
    for col in column_sequence:
        if col.sonar_column is not None:
            target_dtype = table[col.sonar_column].type
            break

    for col in column_sequence:
        if col.text_value is not None:
            vectors, sentences = _get_embed_sentences(col.text_value)
            vectors_extended = pa.chunked_array(
                [pa.ListArray.from_arrays([0, len(vectors)], vectors)] * table_len
            )
            sentences_extended = pa.chunked_array(
                [pa.ListArray.from_arrays([0, len(sentences)], sentences)] * table_len
            )
        else:
            assert (col.text_column is not None) and (col.sonar_column is not None)
            vectors_extended = table[col.sonar_column]
            sentences_extended = table[col.text_column]
            if is_list_like(vectors_extended):
                assert is_list_like(sentences_extended)
            else:
                vectors_extended = simple_array_to_nested(vectors_extended)
                sentences_extended = simple_array_to_nested(sentences_extended)

        if target_dtype and vectors_extended.type != target_dtype:
            vectors_extended = vectors_extended.cast(target_dtype)

        vectors_seq.append(vectors_extended)
        sentences_seq.append(sentences_extended)

    new_vectors_array = hstack_pyarray_list(*vectors_seq)
    new_sentences_array = hstack_pyarray_list(*sentences_seq)
    del vectors_seq, sentences_seq
    table = table.append_column(vector_name, new_vectors_array)
    table = table.append_column(text_name, new_sentences_array)
    return table


@njit
def _get_hierarchical_indices_and_offsets(
    pagaraphs_lengths: List[np.ndarray], max_seq_len: int
):
    indices = []
    new_lens = [0]
    hierarchy_new_lens = [0]

    for i, current_lens in enumerate(pagaraphs_lengths):
        tmp_lens_sum = 0
        nb_blocks = 0
        for ll in current_lens:
            if ll + tmp_lens_sum > max_seq_len:
                indices.append(i)
                new_lens.append(new_lens[-1] + tmp_lens_sum)
                hierarchy_new_lens.append(hierarchy_new_lens[-1] + nb_blocks)

                tmp_lens_sum = ll
                nb_blocks = 0
            else:
                tmp_lens_sum += ll

            nb_blocks += 1

        if nb_blocks > 0:
            indices.append(i)
            new_lens.append(new_lens[-1] + tmp_lens_sum)
            hierarchy_new_lens.append(hierarchy_new_lens[-1] + nb_blocks)

    return (
        np.array(indices, dtype=np.int32),
        np.array(new_lens, dtype=np.int32),
        np.array(hierarchy_new_lens, dtype=np.int32),
    )


def hierarchical_explode_table_with_max_length(
    table: pa.Table,
    columns: Union[str, List[str]],
    max_seq_len: int,
    page_len_column: str,
    page_embs_columns: Optional[Union[str, List[str]]],
) -> pa.Table:
    if isinstance(columns, str):
        columns = [columns]

    if isinstance(page_embs_columns, str):
        page_embs_columns = [page_embs_columns]
    elif page_embs_columns is None:
        page_embs_columns = []

    assert len(columns) > 0

    cols = [pc.fill_null(table[columns[0]], [None])]
    lengths = pc.list_value_length(cols[0]).to_numpy()

    for name in columns[1:]:
        col = pc.fill_null(table[name], [None])
        # checking that all columns list structures are parallel
        assert (lengths == pc.list_value_length(col).to_numpy()).all()
        cols.append(col)

    pagaraphs_lengths = table[page_len_column].to_pandas().to_list()
    # assert [x.sum() for x pagaraphs_lengths] == lengths.tolist()
    # next unroll with max_seq_len
    indices, new_offests, hierarchy_offsets = _get_hierarchical_indices_and_offsets(
        pagaraphs_lengths, max_seq_len
    )

    other_columns = list(table.schema.names)
    for name in set(columns + [page_len_column] + page_embs_columns):
        other_columns.remove(name)

    remaining_table = table.select(other_columns).take(indices)

    result_dict = {}
    for name in other_columns:
        result_dict[name] = remaining_table[name]

    for name, col in zip(columns, cols):
        rolled_array = pa.ListArray.from_arrays(
            offsets=new_offests,
            values=pyarrow_column_to_array(pc.list_flatten(col)),
        )
        result_dict[name] = rolled_array

    for name in set([page_len_column] + page_embs_columns):
        col = table[name]
        rolled_array = pa.ListArray.from_arrays(
            offsets=hierarchy_offsets,
            values=pyarrow_column_to_array(pc.list_flatten(col)),
        )
        result_dict[name] = rolled_array

    return pa.Table.from_pydict(result_dict, schema=table.schema)


def filter_table_with_different_lengths(
    table: pa.Table, columns: List[str]
) -> pa.Table:
    if len(columns) <= 1 or not all(is_list_like(table[col]) for col in columns):
        return table

    ref_lengths = pc.list_value_length(table[columns[0]])
    for col in columns[1:]:
        same_lens = pc.equal(pc.list_value_length(table[col]), ref_lengths)
        if pc.all(same_lens).as_py():
            continue
        else:
            logger.warn(
                f"filtering table whose nb sentences and nb sonar vectors are aligned, keeping {pc.sum(same_lens).as_py()} rows out of{len(table)}"
            )
            table = table.filter(same_lens)
    return table


@dataclass
class PFSState:
    nb_fully_read_files: int = 0
    nb_current_file_read_fragements: int = 0
    total_nb_fragments: int = 0
    total_nb_rows: int = 0


class ParquetFragmentStreamer:
    def __init__(
        self,
        parquet_ds: pq.ParquetDataset,
        split_to_row_groups: bool = True,
        limit_options: Optional[ParquetDatasetLimitOptions] = None,
        read_state: Optional[PFSState] = None,
    ):
        self.split_to_row_groups = split_to_row_groups
        self.limit_options = limit_options or ParquetDatasetLimitOptions()
        self.parquet_ds = parquet_ds

        if read_state is not None:
            self.state = read_state
        else:
            self.reset_state()

    def reset_state(self):
        self.state = PFSState()

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.parquet_ds,
                self.split_to_row_groups,
                self.limit_options,
                self.state,
            ),
        )

    def truncate_files(
        self,
        parquet_ds: pq.ParquetDataset,
        fraction_of_files: Optional[float],
        nb_files: Optional[int],
    ) -> List[pa.dataset.Fragment]:
        file_ds_fragments = get_dataset_fragments(
            parquet_ds, parquet_ds._filter_expression
        )
        self.proxy_ds_path = "/".join(parquet_ds.files[0].split("=")[0].split("/")[:-1])
        logger.info(
            f"{self.proxy_ds_path} : full number of files {len(file_ds_fragments)}"
        )

        if fraction_of_files is not None:
            file_ds_fragments = file_ds_fragments[
                : max(
                    int(round(fraction_of_files * len(file_ds_fragments))),
                    1,
                )
            ]
            logger.info(
                f"{self.proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of fraction_of_files={fraction_of_files}"
            )
        if nb_files is not None and nb_files < len(file_ds_fragments):
            file_ds_fragments = file_ds_fragments[:nb_files]
            logger.info(
                f"{self.proxy_ds_path} : reducing number of files to {len(file_ds_fragments)} because of nb_files={nb_files}"
            )
        return file_ds_fragments

    def __iter__(self):
        limit_options = self.limit_options

        file_ds_fragments = self.truncate_files(
            self.parquet_ds,
            limit_options.fraction_of_files,
            limit_options.nb_files,
        )

        if not self.split_to_row_groups:
            for frag in file_ds_fragments[
                self.state.nb_fully_read_files : limit_options.nb_fragments
            ]:
                self.state.nb_fully_read_files += 1
                yield frag

                if limit_options.nb_rows is not None:
                    self.state.total_nb_rows += frag.count_rows()
                    if self.state.total_nb_rows >= limit_options.nb_rows:
                        break
        else:
            early_stop = False
            logger.info(f"{self.proxy_ds_path} : starting split in row groups")

            for new_file in file_ds_fragments[self.state.nb_fully_read_files :]:
                new_file_fragments = split_fragment_in_row_groups(new_file)
                new_file_fragments = new_file_fragments[
                    self.state.nb_current_file_read_fragements :
                ]
                if limit_options.nb_rows is not None:
                    new_file_fragments_stats = [
                        frag.row_groups[0].num_rows for frag in new_file_fragments
                    ]
                else:
                    new_file_fragments_stats = [0] * len(new_file_fragments)

                for nb_row, frag in zip(new_file_fragments_stats, new_file_fragments):
                    self.state.total_nb_rows += nb_row
                    self.state.total_nb_fragments += 1
                    self.state.nb_current_file_read_fragements += (
                        1  # increate before yield
                    )
                    yield frag

                    if (
                        limit_options.nb_fragments is not None
                        and self.state.total_nb_fragments >= limit_options.nb_fragments
                    ):
                        early_stop = True
                        if limit_options.nb_rows is not None:
                            logger.info(
                                f"{self.proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached with around {self.state.total_nb_rows} rows"
                            )
                        else:
                            logger.info(
                                f"{self.proxy_ds_path} : nb_fragments limit {limit_options.nb_fragments} was reached"
                            )
                        break
                    if (
                        limit_options.nb_rows is not None
                        and self.state.total_nb_rows >= limit_options.nb_rows
                    ):
                        early_stop = True
                        logger.info(
                            f"{self.proxy_ds_path} : nb_rows limit {limit_options.nb_rows} was reached with around {self.state.total_nb_fragments} fragments"
                        )
                        break
                if early_stop:
                    break
                # only when full file is read we increament this
                self.state.nb_fully_read_files += 1
                self.state.nb_current_file_read_fragements = 0


@dataclass
class ShuffledIteratorState:
    epoch_count: int
    current_window: List[Any]
    index: int
    random_state: np.random.RandomState


class ShuffledIterator(Iterator[Any]):
    def __init__(
        self,
        iterator,
        window_size: int,
        nb_epoch: int,
        seed: Optional[int],
        state: Optional[ShuffledIteratorState] = None,
    ):
        self.base_iterator = iterator
        self.window_size = window_size
        self.seed = seed
        self.nb_epoch = nb_epoch

        if state is None:
            state = ShuffledIteratorState(
                random_state=np.random.RandomState(self.seed),
                epoch_count=0,
                current_window=[],
                index=0,
            )
        self.state = state
        self.window_iterator = None

    def reset_state(self):
        self.state.random_state = np.random.RandomState(self.seed)
        self.state.epoch_count = 0
        self._reset_inner()

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.base_iterator,
                self.window_size,
                self.nb_epoch,
                self.seed,
                self.state,
            ),
        )

    def _reset_inner(self):
        self.base_iterator.reset_state()
        self.state.index = 0
        self.state.current_window = []
        self.window_iterator = None

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self.state.epoch_count >= self.nb_epoch:
            raise StopIteration

        # If current window is exhausted, fetch the next window
        if self.window_iterator is None:
            self.window_iterator = batched(self.base_iterator, self.window_size)  # type: ignore
        assert self.window_iterator is not None

        if self.state.index >= len(self.state.current_window):
            try:
                # Get the next window batch
                window = next(self.window_iterator)
                window = np.array(window, dtype="O")
                self.state.random_state.shuffle(window)
                self.state.current_window = window
                self.state.index = 0
            except StopIteration:
                # If no more batches, increment epoch count and reset iterator
                self.state.epoch_count += 1
                self._reset_inner()
                return self.__next__()

        # Return the next element from the current window
        result = self.state.current_window[self.state.index]
        self.state.index += 1
        return result


def stream_parquet_fragments(
    parquet_ds: pq.ParquetDataset,
    nb_epochs: int,
    split_to_row_groups: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = None,
    limit_options: Optional[ParquetDatasetLimitOptions] = None,
    shuffling_window: int = 200,
) -> DataPipelineBuilder:
    fragments_iterator = ParquetFragmentStreamer(
        parquet_ds=parquet_ds,
        split_to_row_groups=split_to_row_groups,
        limit_options=limit_options,
    )

    def reset_fn(iterator):
        iterator.reset_state()
        return iterator

    pipeline = read_iterator(
        ShuffledIterator(
            fragments_iterator,
            window_size=shuffling_window if shuffle else 1,
            nb_epoch=nb_epochs,
            seed=seed,
        ),
        reset_fn,
        infinite=False,
    )

    return pipeline.map(SafeFragment)


def get_row_group_level_metadata(
    dataset: pq.ParquetDataset,
    columns: Optional[List[str]] = None,
    nb_jobs: int = 40,
    max_fragments: int = -1,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Parses row group level metadata from a Parquet dataset and returns it as a pandas DataFrame.
    It's similar to `get_parquet_dataset_metadata`
    but present a unnested view on row groups statistics for only a subset of columns.
    This function can be used for any kind of downstream analysis.

    It uses joblib for parallel processing
    and tqdm for progress tracking, which are good practices for handling large datasets.

    Parameters:
    - dataset (pq.ParquetDataset): The Parquet dataset to parse.
    - columns (list of str, optional): The columns to include in the output DataFrame. If not specified, all columns are included.
                For `columns=[]` no column-vise information will be profided (which is generally much faster).
    - nb_jobs (int, default=40): The number of parallel jobs to run.
    - max_fragments (int, default=-1): The maximum number of fragments to include. If -1, all fragments are included.
    - seed (int, default=123): The seed for the random number generator, used when selecting fragments.

    Returns:
    - pd.DataFrame: A DataFrame containing the row group level metadata.
    Example:
        >>> import pyarrow as pa
        >>> import pyarrow.fs
        >>> import pyarrow.compute as pc
        >>> fs, parquet_uri = pa.fs.FileSystem.from_uri("s3://<bucket_name>/<dataset_name>/")
        >>> dataset = pq.ParquetDataset(parquet_uri, filesystem=fs, filters=pc.equal(pc.field("split"), "validation"))
        >>> df_stats = get_row_group_level_metadata(dataset, columns=["col1", "col2", ...])
    """
    assert max_fragments >= -1
    fragments = list(dataset._dataset.get_fragments(filter=dataset._filter_expression))

    if max_fragments != -1 and max_fragments < len(fragments):
        fragments = (
            np.random.RandomState(seed)
            .choice(np.array(fragments, dtype="O"), max_fragments, replace=False)
            .tolist()
        )

    physical_schema = fragments[0].physical_schema

    columns = columns if columns is not None else physical_schema.names
    # taking only existing columns
    non_existing_columns = tuple(set(columns) - set(physical_schema.names))
    if non_existing_columns:
        print(
            "Following colums are not present in physical schema and will be ignored",
            non_existing_columns,
        )
    columns = [col for col in columns if col in physical_schema.names]

    columns_index = [physical_schema.get_field_index(col) for col in columns]

    columns_to_exclude = set(["row_group_id", "num_rows", "total_byte_size"]) & set(
        columns
    )
    assert len(columns_to_exclude) == 0, (
        f"names conflict, rename/remove : {columns_to_exclude}"
    )

    def get_one_row_group_stats(row_group):
        metadata = row_group.metadata
        info = {
            "row_group_id": row_group.id,
            "num_rows": metadata.num_rows,
            "total_byte_size": metadata.total_byte_size,
        }
        for col, ind in zip(columns, columns_index):
            info[col] = metadata.column(ind).to_dict()
        return info

    def get_fragment_stats(frag):
        return {
            "rg_stats": list(map(get_one_row_group_stats, frag.row_groups)),
            "parquet_file_path": frag.path,
            **get_partition_keys(frag.partition_expression),
        }

    stats = Parallel(nb_jobs, backend="threading")(
        delayed(get_fragment_stats)(frag) for frag in tqdm(fragments)
    )

    stats = pd.DataFrame(stats).explode("rg_stats")
    flatten_row_df = pd.DataFrame(stats.pop("rg_stats").tolist(), index=stats.index)
    result_df = pd.concat([stats, flatten_row_df], axis=1)
    return result_df
