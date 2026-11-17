# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

"""Qwen3 chat format for the Terminus-2 rendering of the Open-SWE outcome-ablation arms."""

# Qwen3 chat format for Terminus-2 conversations, which carry no tool definitions and return every
# observation as a user message. Qwen3's own template drops reasoning from assistant turns before the
# last user message, which here is every turn but the last. This template renders a think block on
# every assistant turn, so each turn is trained as the model generates it after the
# ``<think>\n`` generation prompt. Generation markers supervise reasoning, reply and terminator.
TERMINUS_CHAT_TEMPLATE: str = r"""{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if message.role == 'system' or message.role == 'user' %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- elif '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
        {{- '<|im_start|>assistant\n<think>' }}{% generation %}{{- '\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') + '<|im_end|>' }}{% endgeneration %}{{- '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}
"""
