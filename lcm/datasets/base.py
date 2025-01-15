# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Iterator, Optional, Sequence, TypeVar, Union

import torch
from fairseq2.data.data_pipeline import DataPipeline
from fairseq2.gang import FakeGang, Gang
from fairseq2.typing import DataType

from lcm.datasets.configs import (
    DataLoadingConfig,
    DatasetConfigT,
    create_dataset_config_from_cards,
)
from lcm.datasets.dataloading import (
    build_weighted_pipeline_with_renaming as default_build_fn,
)
from lcm.utils.common import Batched, set_mkl_num_threads

BatchT_co = TypeVar("BatchT_co", bound=Union[Dict, Batched], covariant=True)
logger = logging.getLogger(__name__)


class DataLoader(ABC, Generic[BatchT_co, DatasetConfigT]):
    def __init__(
        self,
        data_config: DataLoadingConfig,
        datasets: Sequence[DatasetConfigT],
        gang: Gang,
        builder_func: Callable[..., DataPipeline] = default_build_fn,
        dtype: DataType = torch.float16,
    ):
        self.data_config = data_config
        self.datasets = list(map(create_dataset_config_from_cards, datasets))
        self.dtype = dtype
        self.gang = gang
        self.builder_func = builder_func

        self._pipeline: Optional[DataPipeline] = None

    @property
    def pipeline(self) -> DataPipeline:
        if self._pipeline is None:
            logger.info(f"R{self.gang.rank} self._pipeline is None, building...")
            gang_rank = self.gang.rank if self.gang else 0
            world_size = self.gang.size if self.gang else 1

            self._pipeline = self.builder_func(
                self.datasets, self.data_config, gang_rank, world_size
            )
        assert self._pipeline, (
            f"Cannot build data pipeline from config {self.data_config}"
        )
        return self._pipeline

    def destroy(self) -> None:
        """Destroy the pipeline to rebuild it with different shuffling"""
        self._pipeline = None
        # Build again and reset it
        logger.info(f"R{self.gang.rank} resetting the pipeline in DataLoader.destroy")
        self.reset()

    def reset(self) -> None:
        """
        Applying reset will result in different shuffling for next iterations,
        since pipeline will use modified generator state from previous one.
        This's suitable side effect for `sharding_in_memory=False` (training) scenario.

        Illustrative example :
        >>> import torch
        >>> from fairseq2.data import read_sequence

        >>> def get_one_epoch_pipeline():
        ...     torch.manual_seed(13)
        ...     return read_sequence(list(range(10))).shuffle(5)

        >>> bb = get_one_epoch_pipeline().and_return()
        >>> list(bb)
        [3, 1, 2, 4, 0, 8, 5, 6, 9, 7]
        >>> bb.reset()
        >>> list(bb)
        [4, 0, 3, 2, 1, 9, 7, 6, 8, 5]
        """
        self.pipeline.reset()

    @abstractmethod
    def iterate_batches(self) -> Iterator[BatchT_co]: ...


class BaseDataLoader(DataLoader[dict, DatasetConfigT]):
    def __init__(
        self,
        data_config: DataLoadingConfig,
        datasets: Sequence[DatasetConfigT],
        dtype: DataType = torch.float16,
        gang: Gang = None,
    ) -> None:
        gang = gang or FakeGang()
        super().__init__(
            data_config=data_config,
            datasets=datasets,
            builder_func=default_build_fn,
            dtype=dtype,
            gang=gang,
        )
        set_mkl_num_threads()

    def iterate_batches(self) -> Iterator[dict]:
        yield from iter(self.pipeline)
