# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

# flake8: noq
import atexit
import bisect
import contextlib
import getpass
import json
import logging
import os
import re
import shlex
import shutil
import socket
import string
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from functools import wraps
from inspect import Parameter
from itertools import accumulate, product
from logging import LogRecord
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import yaml
from numpy.random import RandomState
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm

from lcm.evaluation.api import (
    AverageMetric,
    Example,
    ExampleFn,
    Message,
    MetricFn,
    PredictorConfig,
    Role,
)
from lcm.evaluation.utils.data_utils import as_py, is_tensor

from .distributed import (
    get_dp_rank,
    get_global_rank,
    mp_rank_zero_info,
    mp_rank_zero_warn,
    rank_zero_debug,
)

T = TypeVar("T")
LOGS_CNT = 0


def print_debug_message(msg, logger):
    global LOGS_CNT

    LOGS_CNT += 1
    if LOGS_CNT < 5:
        logger.info(msg)


def _default_json_encoder(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)  # type: ignore
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (date, datetime)):
        return str(o)
    if isinstance(o, (np.float32, np.float16, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if callable(o):
        return repr(o)
    return json.JSONEncoder().default(o)


def write_to_json(
    obj: object,
    path: str,
    mode: str = "w",
    ending: Optional[str] = None,
    **kwargs: Any,
) -> None:
    with open_file(path, mode=mode) as fp:
        json.dump(obj, fp, default=_default_json_encoder, **kwargs)
        if ending is not None:
            fp.write(ending)


def write_to_jsonl(
    items: List[Dict[str, Any]],
    path: str,
    mode: str = "w",
    **kwargs: Any,
) -> None:
    with open_file(path, mode=mode) as fp:
        for item in items:
            fp.write(json.dumps(item, default=_default_json_encoder, **kwargs) + "\n")


def evaluate(
    fn: Callable[..., Any],
    outputs: Union[str, Sequence[str]],
    inputs: Union[str, Sequence[str]] = ("prediction", "targets"),
    collate: bool = False,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    wrapper a native Python function fn to a MetricFn function, e.g. a
    function that applies on an Example and output an Example
    Args:
        outputs: the column names in the output Example that contains
            the results of `fn`
        inputs: The column names in the input Example that `fn` will
            use as parameters
        collate: If set, the input Example is expected to hold a batch
            of input items instead of a single one, while `fn` is a
            function that applies item by item. The direct output of
            `fn` will be collated into the output Example as a dictionary
    """

    @wraps(fn)
    def wrapper(x: Example) -> Example:
        if not collate:
            outputs = fn(*(x[k] for k in input_keys), **kwargs)
        else:
            inputs = zip(*(x[k] for k in input_keys))
            outputs = [fn(*input_, **kwargs) for input_ in inputs]

        if isinstance(outputs, Sequence):
            if len(output_keys) == 1:
                return {output_keys[0]: outputs}
            else:
                if collate:
                    outputs = map(list, zip(*outputs))
                return dict(zip(output_keys, outputs))
        elif isinstance(outputs, Dict):
            return dict(zip(output_keys, outputs.values()))
        else:
            return dict(zip(output_keys, (outputs,)))

    input_keys = (inputs,) if isinstance(inputs, str) else inputs
    output_keys = (outputs,) if isinstance(outputs, str) else outputs
    return wrapper


class ExampleSelector:
    def __init__(
        self,
        examples: Optional[Sequence[Example]] = None,
        num_examples: int = 0,
        select_strategy: Literal["first", "index", "random"] = "first",
        select_indices: Optional[Sequence[int]] = None,
        preprocess_fn: Optional[ExampleFn] = None,
    ) -> None:
        self.examples: Sequence[Example] = examples or []
        self.num_examples: int = num_examples or 0
        assert self.examples is not None and len(self.examples) >= self.num_examples
        self.select_strategy = select_strategy
        self.select_indices = select_indices
        self.preprocess_fn: Optional[ExampleFn] = preprocess_fn

    def __call__(self, random_state: Optional[RandomState] = None) -> List[Example]:
        if self.num_examples == 0:
            return []
        if self.select_strategy == "first":
            examples = self.examples[: self.num_examples]
        elif self.select_strategy == "index":
            assert self.select_indices is not None
            examples = [self.examples[idx] for idx in self.select_indices]
            assert self.num_examples <= len(examples)
            examples = examples[: self.num_examples]
        elif self.select_strategy == "random":
            assert random_state is not None
            indices = random_state.choice(
                len(self.examples), self.num_examples, replace=False
            )
            examples = [self.examples[idx] for idx in indices]

        outputs = deepcopy(examples)
        for x in outputs:
            x.update(self.preprocess_fn(x) if self.preprocess_fn else {})
        return outputs  # type: ignore[return-value]


def string_format(template: str, skip_validation: bool = True, **kwargs: Any) -> str:
    if not skip_validation:
        variables = [k[1] for k in string.Formatter().parse(template) if k[1]]
        if not all(k in kwargs for k in variables):
            raise ValueError(
                f"Expected: {variables}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
            )
        #  `Dict[Optional[str], typing.Any]`.
        kwargs = {k: kwargs[k] for k in variables}
    return template.format(**kwargs)


def filter_by_pattern(names: Iterable[str], pattern: Optional[str]) -> List[str]:
    outputs: List[str] = []
    if pattern is not None:
        for p in pattern.split(","):
            p = p.strip().replace(".", "\\.").replace("*", ".*")
            outputs.extend(filter(re.compile(f"^{p}$").match, names))
    return outputs


def check_pattern(pattern: str, s: str) -> bool:
    match = re.search(pattern, s)
    return bool(match)


def unroll_configs(
    defaults: Mapping[str, Parameter], params: Mapping[str, Any], prefix: str
) -> Dict[str, Dict[str, Any]]:
    def unroll(x: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        x = [product([k], v) if isinstance(v, list) else [(k, v)] for k, v in x.items()]  # type: ignore
        return [dict(item) for item in product(*x)]  # type: ignore

    configs: Dict[str, Dict[str, Any]] = {}
    defaults = {k: v.default for k, v in defaults.items()}
    for kwargs in unroll(params):
        assert kwargs.keys() <= set(defaults.keys()), (kwargs, defaults)
        # Avoid using same name for different task variants
        overrides = {k: v for k, v in kwargs.items() if defaults[k] != v}
        suffix = ".".join(
            f"{k}_{v}" for k, v in overrides.items() if not isinstance(v, (dict, list))
        )
        name = f"{prefix}{'.' + suffix if suffix else ''}"
        configs[name] = {**defaults, **kwargs}
    return configs


def truncate(tokens: List[int], max_length: int, side: str = "left") -> List[int]:
    assert side in ("left", "right")
    return tokens[:max_length] if side == "right" else tokens[-max_length:]


def unroll_chat(messages: List[Message]) -> List[List[Message]]:
    """Always start with system prompt if there is any"""
    # TODO: only works with one system prompt at the moment
    assert messages[0].role == Role.SYSTEM or not any(
        msg.role == Role.SYSTEM for msg in messages
    ), "'unroll_chat' only supports when there is one system prompt at the beginning"
    system_prompt = messages[0] if messages[0].role == Role.SYSTEM else None
    dialogs: List[List[Message]] = []
    reversed_messages = []
    for msg in reversed(messages):
        reversed_messages.append(msg)
        if msg.role == Role.USER:
            dialogs.append(reversed_messages[::-1])
            if system_prompt is not None:
                dialogs.append([system_prompt] + reversed_messages[::-1])
    return dialogs[::-1]


def unroll_msg(content: str) -> List[str]:
    """Returns all list of last words"""
    unroll_msgs: List[str] = [content]
    while True:
        split = content.split(maxsplit=1)
        if len(split) == 2:
            content = split[1]
            unroll_msgs.append(content)
        else:
            if split[0] != content:
                unroll_msgs.append(split[0])
            break
    return unroll_msgs


def truncate_chat(
    messages: List[Message],
    max_prompt_len: int,
    chat_tokenizer_fn: Callable[[List[Message]], List[int]],
) -> List[Message]:
    """
    Left-truncates a dialog so it is the longest sequence of Messages that starts
    with a user message and whose tokenization is below `max_prompt_len`
    TODO: support system role
    """
    assert messages[-1].role == Role.USER
    unrolled_dialogs = sorted(unroll_chat(messages), key=lambda d: -len(messages))
    init_dialog_token_len = 0
    for i, dialog in enumerate(unrolled_dialogs):
        dialog_token_len = len(chat_tokenizer_fn(dialog))
        if i == 0:
            init_dialog_token_len = dialog_token_len
        if dialog_token_len > max_prompt_len:
            continue
        if i > 0:
            mp_rank_zero_info(
                f"Keeping only the last {len(messages)} messages "
                f"and truncating the prompt length from {init_dialog_token_len} to "
                f"{dialog_token_len} < {max_prompt_len} tokens."
            )
        return dialog
    mp_rank_zero_info(
        "Keeping only the last user message "
        f"and truncating the prompt length from {dialog_token_len}"
        f" so it is smaller than to {max_prompt_len} tokens."
    )
    unrolled_last_msg = sorted(unroll_msg(messages[-1].content), key=lambda d: -len(d))
    for i, content in enumerate(unrolled_last_msg):
        dialog_token_len = len(
            chat_tokenizer_fn([Message(role=Role.USER, content=content)])
        )
        if i == 0:
            init_dialog_token_len = dialog_token_len
        if dialog_token_len > max_prompt_len:
            continue
        if i > 0:
            mp_rank_zero_info(
                f"Keeping only the last {len(content.split())} words "
                f"and truncating the prompt length from {init_dialog_token_len} to "
                f"{dialog_token_len} < {max_prompt_len} tokens."
            )
        return dialog
    raise ValueError("max_prompt_len is too small for chat")


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(text: str) -> str:
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def first_answer(text: str, markers: Sequence[str] = ("Q:", "A:")) -> str:
    for marker in markers:
        text = text.split(marker)[0]
    return text


def get_token_offsets(tokenizer: Any, text: str) -> Tuple[List[str], List[int]]:
    from sentencepiece import SentencePieceProcessor  # type: ignore

    if not isinstance(tokenizer, SentencePieceProcessor):
        from tiktoken import Encoding  # type: ignore

        assert isinstance(tokenizer, Encoding)
        token_bytes = tokenizer.decode_tokens_bytes(
            tokenizer.encode(text, allowed_special={"<|reserved_special_token_0|>"})
        )
        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        tokens = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return tokens, offsets
    elif hasattr(tokenizer, "encode_as_immutable_proto"):
        pieces = tokenizer.encode_as_immutable_proto(text).pieces
        tokens = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
    else:
        from sentencepiece import sentencepiece_pb2  # type: ignore

        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(tokenizer.encode_as_serialized_proto(text))
        tokens = [p.surface for p in spt.pieces]
        offsets = list(accumulate((len(t) for t in tokens), initial=0))[:-1]
    return tokens, offsets


def text_index(
    full_text: str, offsets: List[int], text: str, align: str = "right"
) -> slice:
    assert align in ("left", "right")
    start_index = full_text.rfind(text)
    if start_index == -1:
        mp_rank_zero_warn(f"Text '{text}' not found in '{full_text}'")
        return slice(0, 1)
    end_index = start_index + len(text)
    text_start = bisect.bisect_right(offsets, start_index) - 1
    text_end = bisect.bisect_right(offsets, end_index)
    if align == "left":
        return slice(text_start, text_end)
    return slice(text_start - len(offsets) or None, text_end - len(offsets) or None)


def run_metrics(
    x: Example,
    metric_fns: Sequence[MetricFn],
    selected_metrics: Optional[List[str]] = None,
):
    """Run all metrics and update the results back into the example"""
    if selected_metrics:
        fns = [fn for fn in metric_fns if fn.__name__ in selected_metrics]
    else:
        fns = metric_fns  # type: ignore

    x["metrics"] = {k: v for fn in fns for k, v in fn(x).items()}


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record: LogRecord) -> None:
        """Avoid tqdm progress bar interruption by logger's output to console"""
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except RecursionError:
            raise
        except Exception as exc:  # noqa
            self.handleError(record)


def initialize_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    formatter = logging.Formatter(
        f"[%(asctime)s] [rank {get_global_rank()}] [%(levelname)s] %(message)s"
    )
    # stdout: everything
    stdout_handler = TqdmLoggingHandler(sys.stdout)
    stdout_handler.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    stdout_handler.setFormatter(formatter)
    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)
    return logger


def setup_env() -> None:
    triton_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    tiktoken_cache_dir = tempfile.mkdtemp(prefix="/dev/shm/")
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, tiktoken_cache_dir, ignore_errors=True)
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_IB_TIMEOUT": "22",
        "TRITON_CACHE_DIR": triton_cache_dir,
        "TIKTOKEN_CACHE_DIR": tiktoken_cache_dir,
        "AWS_MAX_ATTEMPTS": "10",
        "AWS_RETRY_MODE": "standard",
    }
    for name, value in env_vars.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
            rank_zero_debug(f"WARNING: Setting {name} to {value}")


def get_git_info() -> Dict[str, Any]:
    repo: str = str(Path(__file__).resolve().parent.parent.parent)

    def get_cmd_result(cmd: str, default: str) -> str:
        with contextlib.suppress(Exception):
            result = subprocess.check_output(
                cmd.split(), cwd=repo, stderr=subprocess.DEVNULL
            )
            return result.decode().strip()
        return default

    rev = get_cmd_result("git rev-parse HEAD", "unknown")
    branch = get_cmd_result("git rev-parse --abbrev-ref HEAD", "unknown")
    return {
        "git_repo": repo,
        "commit": rev,
        "branch": branch,
        "user": getpass.getuser(),
    }


def get_open_port() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def set_torch_variables() -> None:
    from torch._utils_internal import TEST_MASTER_ADDR

    os.environ["MASTER_ADDR"] = str(TEST_MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(get_open_port())
    os.environ["TORCHELASTIC_RUN_ID"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


def get_version() -> str:
    from lcm import __version__

    return __version__


def get_gpu_info() -> str:
    mem_stats = torch.cuda.memory_stats()
    usage = {
        "active": mem_stats["active_bytes.all.current"] / 1024**3,
        "allocated": mem_stats["allocated_bytes.all.current"] / 1024**3,
        "reserved": mem_stats["reserved_bytes.all.current"] / 1024**3,
    }
    return ", ".join(f"{k}: {v:.2f}GB" for k, v in usage.items())


def get_random_state(
    seed: int,
    include_data_parallel: bool = True,
    include_job_array: bool = True,
) -> RandomState:
    """
    Construct a random state using a base seed, and optionally, data parallel
    rank and job array task ID.

    Args:
        seed (int): Primary seed value.
        data_parallel_seed (bool): If set, random states are different across DP groups.
        job_array_seed (bool): If set, random states are different across job arrays.
    """
    arr: List[int] = [seed]
    if include_data_parallel:
        arr.append(get_dp_rank())
    if include_job_array:
        arr.append(int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    return RandomState(tuple(arr))


def format_dict(
    data: Union[Dict[str, float], Dict[str, AverageMetric]],
    decimal: int = 6,
    delimiter: str = " | ",
) -> str:
    return delimiter.join(
        f"{k.lower().replace(' ', '_')}: {v.value if isinstance(v, AverageMetric) else v:.{decimal}f}"
        for k, v in data.items()
        if v is not None
    )


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, f"{prefix}{key}/"))
        elif isinstance(value, Tensor):
            assert len(value.size()) == 0, (
                f"Only scalar tensor can be formatted, get {value}"
            )
            flat_dict[f"{prefix}{key}"] = value.item()
        elif isinstance(value, AverageMetric):
            flat_dict[f"{prefix}{key}"] = value.value
        else:
            flat_dict[f"{prefix}{key}"] = value
    return flat_dict


def flatten_and_format(
    data: Dict[str, Any], decimal: int = 6, delimiter: str = " | "
) -> str:
    return delimiter.join(
        f"{k.lower().replace(' ', '_')}: {v:.{decimal}f}" if isinstance(v, float) else v
        for k, v in flatten_dict(data).items()
    )


@contextmanager
def open_file(path: str, mode: str = "r", **kwargs: Any) -> Any:
    if mode == "w" or mode == "a":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    file = open(path, mode=mode, **kwargs)

    try:
        yield file
    finally:
        file.close()


def get_predictor_checkpoint_step(predictor_config: PredictorConfig) -> int:
    """Parse the predictor config and check its checkpoint dir, looking
    for the last step
    """

    def get_checkpoint_step(path, pattern):
        match = re.search(pattern, Path(path).name)
        return int(match.group("step")) if match else -1

    model_dir = getattr(predictor_config, "model_dir", None)
    if model_dir and Path(model_dir).joinpath("checkpoints").exists():
        return get_checkpoint_step(path=model_dir, pattern=r"step_(?P<step>\d+)")

    checkpoint_dir = getattr(predictor_config, "checkpoint_dir", None)
    if checkpoint_dir and Path(checkpoint_dir).exists():
        return get_checkpoint_step(
            path=checkpoint_dir, pattern=r"checkpoint_(?P<step>\d+)"
        )

    model_card = getattr(predictor_config, "model_card", None)
    if not model_card:
        return -1
    elif Path(model_card).exists() and str(model_card).endswith(".yaml"):
        card_path = Path(model_card)
    else:
        package_root_dir = Path(__file__).parent.parent.parent
        card_path = Path(package_root_dir / "card" / (model_card + ".yaml"))
        if not card_path.exists():
            card_path = None
    if card_path:
        with open(card_path, "r", encoding="utf-8") as f:
            card_data = yaml.full_load(f)
            checkpoint_dir = Path(card_data["checkpoint"]).parent
        match = re.search(r"step_(?P<step>\d+)", checkpoint_dir.name)
        return int(match.group("step")) if match else -1
    else:
        return -1


def log_final_results(results, predictor_config, tb_log_dir, metric_log_dir, logger):
    from torch.utils.tensorboard import SummaryWriter

    step = getattr(predictor_config, "step_nr", None)
    if not step:
        step = get_predictor_checkpoint_step(predictor_config)

    if metric_log_dir and get_global_rank() == 0:
        tensorboard_dir = tb_log_dir or os.path.join(metric_log_dir, "tb")
        logger.info(f"Writing Tensorboard logs to {tensorboard_dir}")

        writer = SummaryWriter(tensorboard_dir, max_queue=1000)
        for key, value in results.items():
            if value is not None:
                writer.add_scalar(f"eval/{key}", value, global_step=step)
        writer.close()

        metric_log_path = os.path.join(metric_log_dir, "metrics.eval.jsonl")
        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {"global_step": step, "created_at": datetime.utcnow().isoformat()}
        write_to_json({**timestamp, **results}, metric_log_path, mode="a", ending="\n")  # type: ignore


def log_raw_results(
    raw_results: List[Dict[str, Any]],
    filename: str,
    logger: logging.Logger,
    log_only_text: bool = False,
):
    raw_text_file = filename + ".json"
    raw_tensor_file = filename + ".pt"
    logger.info(f"Writing raw results to {filename} ( *.json | *.pt)")
    raw_text_results: List[Dict[str, Any]] = []
    raw_tensor_results: List[Dict[str, torch.Tensor]] = []
    for raw_result in raw_results:
        json_result = {k: as_py(v) for k, v in raw_result.items() if not is_tensor(v)}
        tensor_result = {k: as_py(v) for k, v in raw_result.items() if is_tensor(v)}
        if json_result:
            raw_text_results.append(json_result)
        if tensor_result:
            raw_tensor_results.append(tensor_result)

    write_to_jsonl(raw_text_results, raw_text_file)
    if not log_only_text and len(raw_tensor_file) > 0:
        torch.save(raw_tensor_results, raw_tensor_file)


def log_config_metadata(cfg, task_name, task_config, logger):
    if cfg.dump_dir is not None:
        # Hydra patch
        if isinstance(cfg, DictConfig):
            _cfg = OmegaConf.to_object(cfg)
        else:
            _cfg = asdict(cfg)

        # remove the launcher info
        assert isinstance(_cfg, dict), f"Unexpected config type: {type(cfg)}"
        _cfg.pop("launcher", None)

        metadata = {
            "timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "command": " ".join(map(shlex.quote, sys.argv)),
            "git_info": get_git_info(),
            "config": _cfg,
            "task_configs": task_config,
        }
        metadata_file = os.path.join(cfg.dump_dir, "metadata.jsonl")
        logger.info(f"Writing configs and metadata to {metadata_file}")
        write_to_json(metadata, metadata_file, mode="a", ending="\n")

    logger.info(f"Evals version {get_version()} ({Path(__file__).parent.parent})")
    logger.info(f"Config: {metadata}")


def parse_omega_list(lst_config: Any) -> Optional[List]:
    """
    robust parsing of omega argumnts, accepting formats like
    [1,2,3,5] OR [1] OR 1 OR 1,2 OR ['src','tgt']
    """
    if lst_config is None:
        return None
    if isinstance(lst_config, (list, ListConfig)):
        lst_value: list = list(lst_config)
    elif isinstance(lst_config, str):
        lst_value = lst_config.strip().strip("[]").split(",")
    else:
        lst_value = [lst_config]
    return lst_value
