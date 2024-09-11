"""The processing module contains functions to format the dataset and build input records."""

from collections.abc import Callable
from copy import deepcopy
from functools import partial

import lxml.html
import weblinx.utils.format as wlf
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from weblinx import (
    Demonstration,
    Replay,
    Turn,
)
from weblinx.processing.dom import clean_and_prune_tree
from weblinx.processing.prompt import (
    find_turns_with_instructor_chat,
    format_utterances,
    get_speaker,
    multi_attempt_format_prev_turns_truncated,
)
from weblinx.processing.truncation import (
    multi_attempt_truncate_cands_turn,
    multi_attempt_truncate_dom_tree,
    reduce_list_of_lengths,
    truncate_text_at_center,
)
from weblinx.utils.recs import get_list_from_records_by_key


def build_formatter_for_multichoice() -> Callable:  # noqa: D103
    format_click = partial(wlf.format_click, formatters=(wlf.format_uid,))
    format_text_input = partial(
        wlf.format_text_input,
        formatters=(
            partial(wlf.format_arg_item, name="text", max_length=200),
            wlf.format_uid,
        ),
    )
    format_change = partial(
        wlf.format_change,
        formatters=(
            partial(wlf.format_arg_item, name="value", max_length=200),
            wlf.format_uid,
        ),
    )
    format_submit = partial(wlf.format_submit, formatters=(wlf.format_uid,))
    format_load = partial(
        wlf.format_load,
        include_transition=False,
        include_timestamp=False,
        max_length=200,
    )
    format_scroll = partial(wlf.format_scroll, include_timestamp=False)

    format_say = partial(wlf.format_say, include_timestamp=False)

    format_intent_auto = partial(
        wlf.format_intent_automatically,
        format_change=format_change,
        format_click=format_click,
        format_load=format_load,
        format_say=format_say,
        format_scroll=format_scroll,
        format_submit=format_submit,
        format_text_input=format_text_input,
    )

    return format_intent_auto


def __get_system_prompt_template_for_llama_mc_concise() -> str:
    sys_prompt_template = (
        "You are an AI assistant with a deep understanding of HTML "
        "and you must predict actions based on a user request, which will be executed. "
        "Use one of the following, replacing [] with an appropriate value: "
        "change(value=[str], uid=[str]) ; "
        "click(uid=[str]) ; "
        "load(url=[str]) ; "
        'say(speaker="navigator", utterance=[str]) ; '
        "scroll(x=[int], y=[int]) ; "
        "submit(uid=[str]) ;"
        "text_input(text=[str], uid=[str]) ;\n"
        "The user's first and last {num_utterances} utterances are: "
        "{utterance_context} ;\n"
        "Viewport size: {height}h x {width}w ;\n"
        "Only the last {num_prev_turns} turns are provided."
    )

    return sys_prompt_template


def __get_candidate_prompt_template_for_llama() -> str:
    return "Here are the top candidates for this turn:\n{candidate_str}"


def __get_final_user_message() -> str:
    return (
        "Please select the best action using the correct format, "
        "do not provide any other information or explanation."
    )


def __merge_prev_turns(prev_turns_text_list: list[str], final_user_message: str) -> list[dict[str, str]]:
    prev_turns_merged: list[dict[str, str]] = []

    for i, turn_text in enumerate(prev_turns_text_list):
        role = get_speaker(
            turn_text,
            instructor_name="user",
            navigator_name="assistant",
            default_name="unknown",
        )

        if i > 0 and prev_turns_merged[-1]["role"] == role:
            prev_turns_merged[-1]["content"] += " " + turn_text
        else:
            prev_turns_merged.append({"role": role, "content": turn_text})

    if len(prev_turns_merged) > 0 and prev_turns_merged[-1]["role"] == "user":
        prev_turns_merged[-1]["content"] += " " + final_user_message
    else:
        prev_turns_merged.append({"role": "user", "content": final_user_message})

    return prev_turns_merged


def __format_utterances_truncated(  # noqa: ANN202, PLR0913
    turns: list["Turn"],
    tokenizer: "PreTrainedTokenizer",
    max_tokens: int,
    format_utterances_fn,
    num_utterances: int = 5,
    type_filter="chat",
    sep=" ",
    convert_to_minutes=True,
    template="[{timestamp}] {utterance}",
    allow_iterative_reduction=False,
):
    utterances = format_utterances_fn(
        turns,
        num_utterances=num_utterances,
        type_filter=type_filter,
        sep=None,
        convert_to_minutes=convert_to_minutes,
        template=template,
    )
    if isinstance(utterances, str):
        utterances = [utterances]

    utterances_str = " ".join(utterances) if sep is None else str(sep).join(utterances)
    utter_tokens = tokenizer.tokenize(utterances_str, add_special_tokens=False)
    num_tokens_to_remove = len(utter_tokens) - max_tokens

    records = []
    for i, text in enumerate(utterances):
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        records.append(
            {
                "index": i,
                "text": text,
                "tokens": tokens,
                "length": len(tokens),
            },
        )

    # NOTE: We only count the token lengths of the values, not the entire formatted string.
    # The full string may have additional tokens. (key, separator, etc.)
    # Consequently, max_total_length is different from max_tokens.
    records = sorted(records, key=lambda r: r["length"])
    lengths_orig = get_list_from_records_by_key(records, "length")  # type: ignore  # noqa: PGH003
    max_total_length = sum(lengths_orig) - num_tokens_to_remove
    lengths_reduced = reduce_list_of_lengths(lengths_orig, max_length=max_total_length)

    for i, rec in enumerate(records):
        red_length = lengths_reduced[i]

        # NOTE: If the length is the same, then we don't need to do anything.
        # Otherwise, we need to truncate the text.
        if red_length >= rec["length"]:
            continue

        trunc = truncate_text_at_center(
            rec["text"],
            tokenizer=tokenizer,
            max_tokens=red_length,
            allow_iterative_reduction=allow_iterative_reduction,
        )

        utterances[rec["index"]] = trunc["text"]

    if sep is None:
        return utterances

    return sep.join(utterances)


def __format_candidates(candidates, max_char_len=300, use_uid_as_rank=False):  # noqa: ANN202
    s = ""
    for cand in candidates:
        doc = cand["doc"].replace("\n", " ").rstrip()
        rank = "uid = " + cand["uid"] if use_uid_as_rank else cand["rank"]

        if max_char_len is not None and len(doc) > max_char_len:
            doc = doc[: max_char_len - 3] + "..."

        s += f"({rank}) {doc}\n"

    return s


def build_prompt_records_for_llama_truncated(  # noqa: D103, PLR0913
    replay: Replay,
    turn: Turn,
    format_intent,
    tokenizer: PreTrainedTokenizer,
    cands_turn=None,
    num_utterances: int = 5,
    num_prev_turns: int = 5,
    system_prompt_template=None,
    candidate_prompt_template=None,
    final_user_message=None,
    include_html=True,
    format_candidates_fn=partial(  # noqa: B008
        __format_candidates,
        max_char_len=None,  # type: ignore  # noqa: PGH003
        use_uid_as_rank=True,
    ),
    merge_prev_turns_fn=__merge_prev_turns,
    format_output_dict_fn: Callable = partial(  # noqa: B008
        wlf.format_output_dictionary,
        function_key="intent",
    ),
    max_html_tokens: int = 700,
    max_utterance_tokens: int = 40 * 5,
    max_prev_turns_tokens: int = 50 * 5,
    max_candidates_tokens: int = 65 * 10,
    add_unused_len_to_cands: bool = True,
    allow_iterative_reduction: bool = False,
    use_tokenizer_template: bool = False,
    template_tokenizer=None,
    parser=None,
) -> list[dict[str, str]]:
    if system_prompt_template is None:
        system_prompt_template = __get_system_prompt_template_for_llama_mc_concise()

    if candidate_prompt_template is None:
        candidate_prompt_template = __get_candidate_prompt_template_for_llama()

    if final_user_message is None:
        final_user_message = __get_final_user_message()

    instructor_chat_turns = find_turns_with_instructor_chat(
        replay,
        turn,
        num_prev_turns=num_prev_turns,
    )
    utterance_context = __format_utterances_truncated(
        instructor_chat_turns,
        tokenizer=tokenizer,
        max_tokens=max_utterance_tokens,
        num_utterances=num_utterances,
        format_utterances_fn=format_utterances,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_text_list = multi_attempt_format_prev_turns_truncated(
        replay=replay,
        turn=turn,
        format_intent=partial(format_intent, return_as=dict),
        tokenizer=tokenizer,
        num_prev_turns=num_prev_turns,
        turn_sep=None,  # type: ignore  # noqa: PGH003
        max_tokens=max_prev_turns_tokens,
        max_attempts=5,
        format_output_dict_fn=format_output_dict_fn,
        warn_after_attempts=False,
        allow_iterative_reduction=allow_iterative_reduction,
    )

    prev_turns_merged = merge_prev_turns_fn(
        prev_turns_text_list=prev_turns_text_list,
        final_user_message=final_user_message,
    )

    sys_prompt = system_prompt_template.format(
        num_utterances=num_utterances - 1,  # NOTE: 1 less since we add the first utterance.
        utterance_context=utterance_context,
        height=turn.viewport_height,
        width=turn.viewport_width,
        num_prev_turns=num_prev_turns,
    )

    if include_html and turn.html not in ["", None] and cands_turn is not None:
        dom_tree_raw = lxml.html.fromstring(turn.html, parser=parser)
        dom_tree_pruned = clean_and_prune_tree(dom_tree_raw, cands_turn=cands_turn)
        trunc = multi_attempt_truncate_dom_tree(
            dom_tree=dom_tree_pruned,
            tokenizer=tokenizer,
            max_tokens=max_html_tokens,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        html = trunc["tree_repr"]
        sys_prompt = html + "\n" + sys_prompt
    else:
        html = ""

    if cands_turn is not None:
        if add_unused_len_to_cands:
            # NOTE: Add the unused length to the candidates.
            num_html_tokens = len(tokenizer.tokenize(html))
            num_utter_tokens = len(tokenizer.tokenize(utterance_context))  # type: ignore  # noqa: PGH003
            if use_tokenizer_template:
                if template_tokenizer is None:
                    msg = "template_tokenizer must be provided when use_tokenizer_template is True."
                    raise ValueError(msg)
                prev_turns_merged_copy = deepcopy(prev_turns_merged)
                if prev_turns_merged[0]["role"] == "assistant":
                    prev_turns_merged_copy.insert(0, {"role": "user", "content": ""})
                num_prev_turns_tokens = len(
                    template_tokenizer.apply_chat_template(
                        [{"role": "system", "content": ""}, *prev_turns_merged_copy],
                        tokenize=True,
                    ),
                )
            else:
                num_prev_turns_tokens = len(
                    tokenizer.tokenize(" ".join(prev_turns_text_list)),
                )
            remain_html_tokens = max_html_tokens - num_html_tokens
            remain_utter_tokens = max_utterance_tokens - num_utter_tokens
            remain_prev_turns_tokens = max_prev_turns_tokens - num_prev_turns_tokens
            remain_tokens = remain_html_tokens + remain_utter_tokens + remain_prev_turns_tokens
            # NOTE: Add the unused length to the max_candidates_tokens.
            max_candidates_tokens += remain_tokens

        cands_turn_trunc = multi_attempt_truncate_cands_turn(
            cands_turn=cands_turn,
            tokenizer=tokenizer,
            max_tokens=max_candidates_tokens,
            format_candidates_fn=format_candidates_fn,
            warn_after_attempts=False,
            allow_iterative_reduction=allow_iterative_reduction,
        )
        cand_str = format_candidates_fn(cands_turn_trunc, max_char_len=None)  # type: ignore  # noqa: PGH003
        cand_prompt = candidate_prompt_template.format(candidate_str=cand_str)
        sys_prompt += cand_prompt[:-1]

    return [{"role": "system", "content": sys_prompt}, *prev_turns_merged]


def __insert_empty_user_content_at_first(prompt: list) -> None:
    if prompt[0]["role"] != "system":
        msg = f"First prompt must be a system prompt. Got {prompt[0]['role']} instead."
        raise ValueError(msg)

    if prompt[1]["role"] != "user":
        prompt.insert(1, {"role": "user", "content": ""})


def insert_formatted_chat_into_records(  # noqa: D103
    records,
    demos: list[Demonstration],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    *,
    include_output_target: bool = True,
) -> list:
    processed_records = deepcopy(records)
    for i, record in enumerate(records):
        __insert_empty_user_content_at_first(record["prompt"])

        if include_output_target:
            target = [{"role": "assistant", "content": record["output_target"]}]
            combined = record["prompt"] + target
        else:
            combined = record["prompt"]

        # NOTE: The `apply_chat_template` method of the tokenizer is required.
        text = str(
            tokenizer.apply_chat_template(
                combined,
                tokenize=False,
                add_generation_prompt=False,
            ),
        )

        processed_records[i]["text"] = text

        processed_records[i]["tasks"] = next(
            filter(lambda demo: demo.form["shortcode"] == record["demo_name"], demos),
        ).form["tasks"]

    return processed_records
