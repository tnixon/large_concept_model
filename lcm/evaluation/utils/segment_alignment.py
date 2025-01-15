# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

import typing as tp
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import numpy as np
import pandas as pd
import spacy
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from tqdm.auto import tqdm


def get_all_ngrams(doc, min_n=1, max_n=3):
    words = [str(x) for x in doc]
    result = list()
    for n in range(min_n, max_n + 1):
        for i in range(0, len(words) - n + 1):
            result.append(str(doc[i : i + n]))
    return result


def monotonic_align(sims: np.ndarray) -> tp.List[tp.Tuple[int, int]]:
    """
    Given an array of similarity values, compute a strictly monotonic path (possibly with skips)
    with the maximal sum of similarities along the path.
    Skipping happens if the similaritie are negative, so they would otherwise decrease the total.
    """
    nrows, ncols = sims.shape

    rewards = np.zeros_like(sims)
    choices = np.zeros_like(sims).astype(
        int
    )  # 1: choose this pair, 2: decrease i, 3: decrease j

    for i in range(nrows):
        for j in range(ncols):
            # Option 1: align i to j
            score_add = sims[i, j]
            if i > 0 and j > 0:
                score_add += rewards[i - 1, j - 1]
                choices[i, j] = 1
            best = score_add
            # Option 2: skip i, align j to the best alignment before
            if i > 0 and rewards[i - 1, j] > best:
                best = rewards[i - 1, j]
                choices[i, j] = 2
            # Option 3: skip j, align i to the best alignment before
            if j > 0 and rewards[i, j - 1] > best:
                best = rewards[i, j - 1]
                choices[i, j] = 3
            rewards[i, j] = best

    # backtracking the optimal alignment
    alignment = []
    i = nrows - 1
    j = ncols - 1
    while i >= 0 and j >= 0:
        if choices[i, j] in {
            0,
            1,
        }:  # 0 occurs only in the pair of first sentences, if we are at it
            alignment.append((i, j))
            i -= 1
            j -= 1
        elif choices[i, j] == 2:
            i -= 1
        else:
            j -= 1
    return alignment[::-1]


def _get_cost(i, j, sims, r):
    # the cost of splitting immediately before the (i, j) pair
    prev2next = sims[max(0, i - r) : i, j : j + r]
    next2prev = sims[i : i + r, max(0, j - r) : j]
    # print(next2prev.shape, prev2next.shape)
    # plt.imshow(sims[max(0,i-r):i+r, max(0, j-r):j+r])
    # plt.figure()
    # plt.imshow(prev2next)
    denominator = prev2next.numel() + next2prev.numel()
    if denominator == 0:
        return 0
    return (prev2next.sum() + next2prev.sum()).item() / denominator


def convert_multisentence_segments(
    text: str, sentences: List[str], segments_start_end_ids: List[Tuple[int, int]]
) -> List[str]:
    """
    Given a document, a list of all sentences in the document, and a list of segment start and end sentence ids, split the text into segments.
    The output is non-trivial, because we want to preserve the separators between sentences (e.g. whitespace or newline).
    """
    src_sents_starts = []
    start = 0
    for sent in sentences:
        start = text.index(sent, start)
        src_sents_starts.append(start)
    src_sents_starts.append(len(text))

    text_segments = [
        text[src_sents_starts[start_id] : src_sents_starts[end_id]]
        for start_id, end_id in segments_start_end_ids
    ]
    return text_segments


def bipartite_split(
    similarities: torch.Tensor,
    alignment: List[Tuple[int, int]],
    max_len: int = 100,
    min_src_len: int = 10,
    min_tgt_len: int = 1,
    cost_radius: int = 10,
    unequal_split_penalty_power: float = 1.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Based on the target-source sentence similarities and their alignment, produce a set of segments that would be aligned as well.
    Return two lists of tuples: one for the target segments start and next-after-last sentence ids, and one for the source segments.
    """
    # Part 1: create potential alignment-respecting splits and compute their costs
    cand_splits = [(0, 0)]
    cand_split_costs = [100]
    prev_i, prev_j = -1, -1
    for i, j in alignment:
        if i == prev_i + 1 and j == prev_j + 1:
            best_cand = (i, j)
        else:
            best_cand = (-1, -1)
            best_cost = 100
            for i_cand in range(prev_i + 1, i + 1):
                for j_cand in range(prev_j + 1, j + 1):
                    cost = _get_cost(i_cand, j_cand, similarities, cost_radius)
                    if cost < best_cost:
                        best_cand = (i_cand, j_cand)
                        best_cost = cost
            # TODO: find the optimal split among multiple candidates
        if best_cand != (0, 0):
            cand_splits.append(best_cand)
            cand_split_costs.append(
                _get_cost(best_cand[0], best_cand[1], similarities, cost_radius)
            )
        prev_i, prev_j = i, j

    cand_splits.append((similarities.shape[0], similarities.shape[1]))
    cand_split_costs.append(100)

    # Part 2: do the recursive splitting until we cover all the segments
    # each segment is characterized by the ids of its first and next-after-last points.
    unsplit_segments_backward = [(0, len(cand_splits) - 1)]
    segments = []
    while unsplit_segments_backward:
        first_id, last_id = unsplit_segments_backward.pop()
        first, last = cand_splits[first_id], cand_splits[last_id]
        # if the segment is short enough, just move it to the finalized list
        if last[0] - first[0] <= max_len and last[1] - first[1] <= max_len:
            segments.append((first_id, last_id))
            continue

        # generate the possible splits
        possible_splits = []
        for mid_id in range(first_id + 1, last_id):
            mid = cand_splits[mid_id]
            if mid[0] - first[0] < min_src_len or mid[1] - first[1] < min_tgt_len:
                continue
            if last[0] - mid[0] < min_src_len or last[1] - mid[1] < min_tgt_len:
                continue
            left_len, right_len = mid[0] - first[0], last[0] - mid[0]
            penalty = (
                max(left_len, right_len) / min(left_len, right_len)
            ) ** unequal_split_penalty_power

            possible_splits.append((cand_split_costs[mid_id] * penalty, mid, mid_id))

        # if there is nothing to split, add the segment as is, even if it is too long
        if len(possible_splits) == 0:
            segments.append((first_id, last_id))
            continue

        # choose the best split point out of the available candidates
        possible_splits = sorted(possible_splits)
        mid_cost, mid, mid_id = possible_splits[0]
        # print(first_id, last_id, "=>", mid_id, mid_cost)

        # add the new segments to the stack (in the reverse order)
        unsplit_segments_backward.append((mid_id, last_id))
        unsplit_segments_backward.append((first_id, mid_id))

    # convert the segments to sentence ids

    tgt_segments_ids = [
        (cand_splits[first_id][0], cand_splits[last_id][0])
        for first_id, last_id in segments
    ]
    src_segments_ids = [
        (cand_splits[first_id][1], cand_splits[last_id][1])
        for first_id, last_id in segments
    ]
    return tgt_segments_ids, src_segments_ids


def match_text_pairs(left_texts: List[str], right_texts: List[str]) -> List[int]:
    """
    For each text on the left side, find the id of the most similar text on the right side.
    The texts are compared as bags of word n-grams; for long texts, this is much faster than edit distance.
    """

    def text2vec(text):
        return Counter(get_all_ngrams(text.split(), 1, 2))

    def bag_diff(b1, b2):
        return sum(abs(b1[k] - b2[k]) for k in set(b1.keys()).union(b2.keys()))

    left_bags = [text2vec(t) for t in left_texts]
    right_bags = [text2vec(t) for t in right_texts]
    results = []
    for bag in left_bags:
        distances = [bag_diff(bag, ref_bag) for ref_bag in right_bags]
        closest = min(distances)
        match = [i for i, d in enumerate(distances) if d <= closest]
        assert len(match) == 1
        results.extend(match)
    return results


def split_and_embed(
    text, encoder, spacy_model, lang: str = "eng_Latn"
) -> Tuple[List[str], torch.Tensor]:
    doc = spacy_model(text)
    sents = [str(sent) for sent in doc.sents]
    embs = encoder.predict(sents, source_lang=lang)
    return sents, embs


def align_outputs_with_paragraphs(
    root_dir: str,
    outputs_filename: str,
    paragraphs_dirname: str,
    result_filename: str = "paragraph_aligned_outputs.csv",
    output_column_prefix: str = "Summary",
    input_src_text_column: str = "content",
    output_src_text_column: str = "text",
    result_column_suffix: str = "_by_paragraph",
    device_name: str = "cuda",
):
    """
    Read the machine outputs (summaries etc) from the `outputs_filename` and paragraph-split inputs
    from per-document files in `paragraphs_dirname`; align them on the paragraph level, and save to `result_filename`.
    """
    # 1. Extract the input data.

    # 1.1 Read the machine summarization outputs
    output_df = pd.read_csv(Path(root_dir) / outputs_filename, index_col=0)
    machine_inputs = output_df[output_src_text_column].tolist()

    output_columns = [
        c for c in output_df.columns if c.startswith(output_column_prefix)
    ]
    assert len(output_columns) > 0, (
        f"No output columns starting with the output_column_prefix='{output_column_prefix}' found."
    )

    # 1.2. Read the human alignments
    paragraphs_ids = []
    paragraphs_data = []

    for fn in sorted((Path(root_dir) / paragraphs_dirname).iterdir()):
        if fn.suffix != ".csv":
            continue
        paragraphs_ids.append(fn.stem)
        df = pd.read_csv(fn)
        df["document_file_stem"] = fn.stem
        paragraphs_data.append(df)

    paragraphs_texts = [
        "\n\n".join(df[input_src_text_column].tolist()) for df in paragraphs_data
    ]

    # 1.3. The order of the documents might be different. Align it!
    assert len(machine_inputs) == len(paragraphs_texts), (
        f"Machine outputs contain {len(machine_inputs)} documents, but human paragraphs contain {len(paragraphs_texts)} documents."
    )
    machine_ids = match_text_pairs(paragraphs_texts, machine_inputs)
    assert len(machine_ids) == len(set(machine_ids)), (
        "Machine and human outputs don't seem to be 1:1 alignable."
    )

    named_machine_outputs: Dict[str, List[str]] = {
        column: output_df[column].loc[machine_ids].tolist() for column in output_columns
    }

    # 2. Process the data
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device_name),
    )
    spacy_model = spacy.load("en_core_web_md")

    for doc_idx, df in enumerate(tqdm(paragraphs_data)):
        # 2.1. sonarize the source
        src_texts = df[input_src_text_column].tolist()
        src_embs_list, src_sents_list = [], []
        for src_text in src_texts:
            src_sents, src_embs = split_and_embed(src_text, t2vec_model, spacy_model)
            src_embs_list.append(src_embs)
            src_sents_list.append(src_sents)
        src_embs_all = torch.concat(src_embs_list)
        src_sents_all = []  # [sent for chunk in src_sents_list for sent in chunk]
        src_sent_paragraph_ids = []
        for paragraph_id, paragraph_sents in enumerate(src_sents_list):
            src_sents_all.extend(paragraph_sents)
            src_sent_paragraph_ids.extend([paragraph_id] * len(paragraph_sents))

        for target_name, target_values in named_machine_outputs.items():
            # 2.2. sonarize the targets
            tgt_text = target_values[doc_idx]
            tgt_sents, tgt_embs = split_and_embed(tgt_text, t2vec_model, spacy_model)

            # 2.3 compute the sentence alignment
            tgt_embs_normed = torch.nn.functional.normalize(tgt_embs)
            src_embs_normed = torch.nn.functional.normalize(src_embs_all)
            tgt2src_sim = tgt_embs_normed.matmul(src_embs_normed.T)
            alignment = monotonic_align(tgt2src_sim.cpu().numpy())
            # 2.4 For each target sentence, assign it to the source paragraph with which it aligns the most frequently
            tgt_sent_id_to_paragraphs: tp.DefaultDict[int, tp.Counter[int]] = (
                defaultdict(Counter)
            )
            for tgt_sent_id, src_sent_id in alignment:
                tgt_sent_id_to_paragraphs[tgt_sent_id][
                    src_sent_paragraph_ids[src_sent_id]
                ] += 1

            # 2.5 for each source paragraph, get the target sentences that have been assigned to it
            src_paragraph_id_to_tgt_sentences = defaultdict(list)
            for tgt_sent_id, paragraph_counter in tgt_sent_id_to_paragraphs.items():
                paragraph_id = paragraph_counter.most_common(1)[0][0]
                src_paragraph_id_to_tgt_sentences[paragraph_id].append(tgt_sent_id)

            src_paragraphs_start_end_ids = []
            for paragraph_id in range(len(src_texts)):
                tgt_ids = src_paragraph_id_to_tgt_sentences[paragraph_id]
                if len(tgt_ids) == 0:
                    src_paragraphs_start_end_ids.append((0, 0))
                else:
                    src_paragraphs_start_end_ids.append(
                        (min(tgt_ids), max(tgt_ids) + 1)
                    )

            # 2.6 for each source paragraph, contacenate the corresponding target sentences back to a "target paragraph"
            tgt_segments = convert_multisentence_segments(
                tgt_text, tgt_sents, src_paragraphs_start_end_ids
            )
            # 2.7 add it to the paragraph-segmented dataframe
            paragraphs_data[doc_idx][target_name + result_column_suffix] = tgt_segments

    # 3. Save the results to disk
    result_df = pd.concat(paragraphs_data)
    result_df.to_csv(Path(root_dir) / result_filename)
    return result_df


if __name__ == "__main__":
    fire.Fire(align_outputs_with_paragraphs)
