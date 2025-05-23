# # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#


import asyncio
from pathlib import Path

import fire
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextEmbedderConfig,
    SonarTextBatchEmbedder
)
from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.hf_shards import HFInputConfig
from stopes.utils.sharding.parquet_shards import (
    ParquetOutputConfig,
)

from lcm.datasets.sentence_splitter_pipeline import (
    FullPipeline,
    FullPipelineConfig,
    SentenceSplitterConfig,
)


# Create a modified version of the SonarTextBatchEmbedder class
class DistributedSonarTextBatchEmbedder(SonarTextBatchEmbedder):
    def __init__(self, config):
        # Make a copy of the config to modify
        self.original_config = config
        # We'll set the device in __call__ based on the worker ID
        super().__init__(config)
    
    def __call__(self, batch):
        # Get the worker ID and assign a GPU
        import os
        import torch
        
        # Get worker ID from environment or use process ID as fallback
        worker_id = int(os.environ.get("SLURM_PROCID", os.getpid() % 8))
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 0:
            # Assign GPU based on worker ID
            gpu_id = worker_id % num_gpus
            device = f"cuda:{gpu_id}"
            
            # Move models to the assigned GPU
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder.to(device)
            
            print(f"Worker {worker_id} using device {device} for SONAR embedding")
        
        # Call the parent method
        return super().__call__(batch)


# Create a modified version of FullPipeline to use our distributed embedder
class DistributedFullPipeline(FullPipeline):
    def __init__(self, config: FullPipelineConfig):
        super().__init__(config)
        # overrride the encoder with our new implementation
        self.sonar_encoder = DistributedSonarTextBatchEmbedder(self.config.sonar_encoder_config)



def run(output_dir: Path):
    """
    launch a preprocessing pipeline, this will use SAT to split text in sentences and then use SONAR to
    embed each sentence.
    This example downloads data from huggingface and outputs it to a parquet dataset.

    `output_dir` is the directory where the processed data will be written. The output will be in a parquet file format.
    """
    # setup the sentence splitter
    splitter_config = SentenceSplitterConfig(
        columns=["text"],  # this is the column in the input dataset where we expect to find text to split
        model_name="sat-3l",
        verbose=True,
        sentence_threshold=0.2,  # sentence splitting threshold to tune based on the data (domain, language, etc.)
        max_sentence_len=256,
    )
    # setup SONAR, we are only going to deal with english
    sonar_encoder_config = SonarTextEmbedderConfig(
        column_config=[  # we can process several columns at once which is useful for finetuning datasets
            LangColumnConfig("text_sentences", lang_value="eng_Latn")
        ],  # splitter has output a new column `text_sentences` and this is what we will embed
    )
    # setup the full pipeline, that will use the splitter and the sonar embeddings,
    full_config = FullPipelineConfig(
        splitter_config=splitter_config,
        sonar_encoder_config=sonar_encoder_config,
    )

    # setup the input to download from huggingface, adjust this to the dataset you care about
    # Checkout https://github.com/facebookresearch/stopes/tree/main/stopes/utils/sharding for other potential
    # input systems (jsonl, parquet) and how to configure them in this pipeline.
    input_config = HFInputConfig(
        input_file="wikimedia/wikipedia",
        data_dir="20231101.en",
        split="train[0:200]",  # we are only taking a small sample for the toy example
        num_shards=1,  # as we have a small sample, we don't need many shards, you should increase this for larger datasets
        batch_format=BatchFormat.ARROW,
        batch_size=5,  # adjust to your system's size
    )
    # input_config = HFInputConfig(
    #     input_file="wikimedia/wikipedia",
    #     data_dir="20231101.en",
    #     split="train",  # we are only taking a small sample for the toy example
    #     num_shards=8*10,  # as we have a small sample, we don't need many shards, you should increase this for larger datasets
    #     batch_format=BatchFormat.ARROW,
    #     batch_size=8,  # adjust to your system's size
    # )
    # input_config = HFInputConfig(
    #     input_file="HuggingFaceFW/fineweb-edu",
    #     data_dir="sample-100BT",
    #     split="train",  # we are only taking a small sample for the toy example
    #     num_shards=64,  # as we have a small sample, we don't need many shards, you should increase this for larger datasets
    #     batch_format=BatchFormat.ARROW,
    #     batch_size=8,  # adjust to your system's size
    # )
    # setup the output to write to parquet
    output_config = ParquetOutputConfig(
        output_dir,
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )

    # requirements for our slurm jobs, if you are using a local cpu, you can ignore this
    # if you are using slurm but no gpus, remove the gpus_per_node config
    # req = Requirements(
    #     nodes=1,
    #     tasks_per_node=8,
    #     mem_gb=80, 
    #     gpus_per_node=8, 
    #     cpus_per_task=10, 
    #     timeout_min=3 * 24 * 60
    # )
    req = Requirements(
        mem_gb=120, 
        gpus_per_node=1, 
        cpus_per_task=10, 
        timeout_min=3 * 24 * 60
    )
    # launching config, here we use `local` to run locally, but you can switch it to `slurm` if you have a SLURM cluster.
    launcher = Launcher(
        cache=None,
        cluster="local",
        # for SLURM you can set some parameters of the launcher here
        # cluster="slurm",
        # update_parameters={
        #    "partition": "learn",
        # },
    )

    # launch the shards processing
    # stopes_wrapped = stopes_data_mapper(req, {"name": "prep_wiki_full"})(FullPipeline)
    stopes_wrapped = stopes_data_mapper(req, {"name": "prep_wiki"})(DistributedFullPipeline)
    stopes_module = stopes_wrapped(input_config, output_config, full_config)

    asyncio.run(launcher.schedule(stopes_module))


if __name__ == "__main__":
    fire.Fire(run)
