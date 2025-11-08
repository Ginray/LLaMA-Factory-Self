# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, TypedDict

from typing_extensions import NotRequired, Required

from ...extras.types import DPOSample, Sample, SFTSample
from ...extras import logging

logger = logging.get_logger(__name__)


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: Required[str]
    input: NotRequired[str]
    output: Required[str]
    history: NotRequired[str]


ShareGPTMessage = TypedDict("ShareGPTMessage", {
    "from": Required[str],  # Role of the message sender (e.g., "human", "gpt", "system")
    "value": Required[str]  # Content of the message
})


class ShareGPTSample(TypedDict, total=False):
    """Type definition for raw ShareGPT sample."""
    conversations: Required[list[ShareGPTMessage]]
    id: NotRequired[str]
    meta: NotRequired[dict]


class PairSample(TypedDict, total=False):
    prompt: NotRequired[str]
    chosen: NotRequired[list[dict]]
    rejected: NotRequired[list[dict]]


def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """Convert Alpaca sample to SFT sample.

    Args:
        raw_sample (AlpacaSample): Alpaca sample.

    Returns:
        SFTSample: SFT sample.
    """
    messages = []
    if "system" in raw_sample:
        messages.append(
            {"role": "system", "content": [{"type": "text", "value": raw_sample["system"]}], "loss_weight": 0.0}
        )

    if "history" in raw_sample and isinstance(raw_sample["history"], list):
        for old_prompt, old_response in raw_sample["history"]:
            messages.append(
                {"role": "user", "content": [{"type": "text", "value": old_prompt}], "loss_weight": 0.0}
            )

            messages.append(
                {"role": "assistant", "content": [{"type": "text", "value": old_response}], "loss_weight": 1.0}
            )

    if "instruction" in raw_sample or "input" in raw_sample:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": raw_sample.get("instruction", "") + raw_sample.get("input", "")}
                ],
                "loss_weight": 0.0,
            }
        )

    if "output" in raw_sample:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "value": raw_sample["output"]}], "loss_weight": 1.0}
        )

    return {"messages": messages}


def sharegpt_converter(raw_sample: ShareGPTSample) -> SFTSample:
    """
    Converts a raw ShareGPT sample into a formatted SFT (Supervised Fine-Tuning) sample.
    The logic of this function is consistent with the v0 version, while only retaining the SFT scenarios.

    Args:
        raw_sample (ShareGPTSample): A raw sample in ShareGPT format.

    Returns:
        dict: A dictionary containing the formatted 'messages' list for SFT training.
              Returns an empty list if the input data is invalid.
    """
    tag_mapping = {
        "human": "user",
        "gpt": "assistant",
        "observation": "observation",
        "function_call": "function",
    }
    odd_tags = ("human", "observation")
    even_tags = ("gpt", "function_call")
    accept_tags = (odd_tags, even_tags)
    messages = raw_sample["conversations"]
    aligned_messages = []
    system_content = ""

    # Extract and handle system message if present (typically the first message)
    if len(messages) != 0 and messages[0]["from"] == "system":
        system_content = messages[0]["value"]
        messages = messages[1:]
    else:
        system_content = ""

    aligned_messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "value": system_content}],
            "loss_weight": 0.0
        }
    )

    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message["from"] not in accept_tags[turn_idx % 2]:
            logger.warning_rank0(f"Invalid role tag in {messages}.")
            broken_data = True
            break

        aligned_messages.append(
            {
                "role": tag_mapping[message["from"]],
                "content": [{"type": "text", "value": message["value"]}],
                "loss_weight": 0.0 if message["from"] in odd_tags else 1.0
            }
        )

    if len(aligned_messages) % 2 == 0:  # The count after including the system message must be an odd number.
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if broken_data:
        logger.warning_rank0("Skipping this abnormal example.")
        return {"messages": []}
    else:  # normal example
        return {"messages": aligned_messages}


def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to standard DPO sample.

    Args:
        raw_sample (PairSample): pair sample with prompt, chosen, rejected fields.
        see raw example at: https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
        see the standard DPO sample at: https://huggingface.co/datasets/frozenleaves/v1-dpo-demo/raw/main/v1-dpo-demo.jsonl
    """
    chosen_messages = []
    assert "chosen" in raw_sample, "chosen field is required in pair sample."
    assert "rejected" in raw_sample, "rejected field is required in pair sample."
    assert isinstance(raw_sample["chosen"], list) and isinstance(raw_sample["rejected"], list), (
        "chosen and rejected field should be a list[dict], or you may need to implement your custom converter."
    )

    if "chosen" in raw_sample:
        value = raw_sample.get("chosen", "")
        for item in value:
            if item.get("role", "") == "system":
                chosen_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "user":
                chosen_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "assistant":
                chosen_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 1.0,
                    }
                )

    rejected_messages = []
    if "rejected" in raw_sample:
        value = raw_sample.get("rejected", "")
        for item in value:
            if item.get("role", "") == "system":
                rejected_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "user":
                rejected_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 0.0,
                    }
                )
            if item.get("role", "") == "assistant":
                rejected_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "value": item.get("content", "")}],
                        "loss_weight": 1.0,
                    }
                )

    return {"chosen_messages": chosen_messages, "rejected_messages": rejected_messages}


CONVERTERS = {
    "alpaca": alpaca_converter,
    "pair": pair_converter,
    "sharegpt": sharegpt_converter,
}


def get_converter(converter_name: str) -> Callable[[dict], Sample]:
    if converter_name not in CONVERTERS:
        raise ValueError(f"Converter {converter_name} not found.")

    return CONVERTERS[converter_name]
