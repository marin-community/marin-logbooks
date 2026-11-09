# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count close-related patterns in each exported 10.05 SFT batch (trained turns only).

Usage: scan_batches.py OUT_JSONL FILES...
"""
import io
import itertools
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zstandard
from transformers import AutoTokenizer

BOT, START_THINK, END_THINK, HDR, END_HDR, EOM, EOT = 128000, 128002, 128003, 128006, 128007, 128008, 128009
THINK_WORD = 27963
tok = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
vocab = tok.convert_ids_to_tokens(list(range(128000)))
dec = [tok.convert_tokens_to_string([v]) if v else "" for v in vocab]
ENDS_LTS = np.array([i for i, s in enumerate(dec) if s.endswith("</")])
ENDS_LT = np.array([i for i, s in enumerate(dec) if s.endswith("<")])
STARTS_GT = np.array([i for i, s in enumerate(dec) if s.startswith(">")])
ASSISTANT = tuple(tok.encode("assistant", add_special_tokens=False))
EVENT_IDS = np.array([BOT, START_THINK, END_THINK, HDR, END_HDR, EOM, EOT])


def load(path):
    a = np.load(io.BytesIO(zstandard.ZstdDecompressor().decompressobj().decompress(Path(path).read_bytes())))
    return a["tokens"].astype(np.int64), a["loss_weight"], a["segment_ids"]


def scan(path):
    t, w, seg = load(path)
    c = Counter()
    c["trained_tokens"] = int((w > 0).sum())
    for r in range(t.shape[0]):
        row, wr, sr = t[r], w[r], seg[r]
        # plain-text tag positions
        cand = np.nonzero(row[1:-1] == THINK_WORD)[0] + 1
        cand = cand[np.isin(row[cand + 1], STARTS_GT)]
        txt_close = set(cand[np.isin(row[cand - 1], ENDS_LTS)].tolist())
        txt_open = set(cand[np.isin(row[cand - 1], ENDS_LT)].tolist())
        pos = np.nonzero(np.isin(row, EVENT_IDS))[0].tolist()
        events = sorted(
            [(p, int(row[p])) for p in pos] + [(p, "txt_close") for p in txt_close] + [(p, "txt_open") for p in txt_open]
        )
        role, hdr_at, turn, cur_seg = None, None, None, None

        def finish(turn, end=None):
            if turn is None or not turn["trained"]:
                return
            end = turn["last"] if end is None else end
            if turn["close_at"] is not None:
                c["post_close_tokens"] += end - turn["close_at"]
            elif "O" not in turn["seq"]:
                c["nothink_asst_tokens"] += end - turn["begin"]
            c["asst_turns"] += 1
            k = turn["seq"]
            n_open, n_close = k.count("O"), k.count("C")
            c["turn_has_think"] += n_open > 0
            c["turn_closed_once"] += n_open == 1 and n_close == 1 and k.index("O") < k.index("C")
            c["turn_open_unclosed"] += n_open > 0 and (n_close == 0 or k.rindex("O") > k.rindex("C"))
            c["turn_orphan_close"] += n_close > 0 and (n_open == 0 or k.index("C") < k.index("O"))
            c["turn_ge2_close"] += n_close >= 2
            c["turn_reopen_after_close"] += "C" in k and "O" in k[k.index("C") :]
            c["turn_close_close_no_open_between"] += any(a == "C" and b == "C" for a, b in itertools.pairwise(k))
            c["closes_trained"] += n_close
            # plain-text tags relative to the special close
            first_c = k.find("C")
            c["turn_txt_close_in_think"] += any(x == "x" for x in (k[:first_c] if first_c >= 0 else k))
            c["turn_txt_close_after_close"] += first_c >= 0 and "x" in k[first_c:]
            c["turn_txt_open_after_close"] += first_c >= 0 and "o" in k[first_c:]
            c["turn_txt_close_no_special"] += n_open == 0 and n_close == 0 and "x" in k
            c["think_tokens"] += turn["think_tokens"]

        for p, e in events:
            if sr[p] != cur_seg:
                finish(turn)
                turn, role, cur_seg = None, None, sr[p]
            if e == HDR:
                if turn is not None:
                    c["asst_turn_no_eot"] += 1
                finish(turn, p)
                turn = None
                hdr_at = p
            elif e == END_HDR and hdr_at is not None:
                role = tuple(row[hdr_at + 1 : p].tolist())
                if role == ASSISTANT:
                    turn = {
                        "seq": "",
                        "trained": bool(wr[p + 1 : p + 2].any()),
                        "think_tokens": 0,
                        "open_at": None,
                        "close_at": None,
                        "begin": p,
                        "last": p,
                    }
                else:
                    turn = None
                    if role != tuple(tok.encode("user", add_special_tokens=False)) and role != tuple(
                        tok.encode("system", add_special_tokens=False)
                    ):
                        c["other_role_turns"] += 1
                hdr_at = None
            elif turn is not None:
                turn["last"] = p
                if e in (EOT, EOM, BOT):
                    if e == EOM:
                        c["eom_turn_ends"] += 1
                    finish(turn, p)
                    turn = None
                elif e == START_THINK:
                    turn["seq"] += "O"
                    turn["open_at"] = p
                elif e == END_THINK:
                    turn["seq"] += "C"
                    if turn["close_at"] is None:
                        turn["close_at"] = p
                    if turn["open_at"] is not None:
                        turn["think_tokens"] += p - turn["open_at"]
                        turn["open_at"] = None
                elif e == "txt_close":
                    turn["seq"] += "x"
                elif e == "txt_open":
                    turn["seq"] += "o"
            else:
                if e == START_THINK:
                    c["start_think_outside_asst"] += 1
                if e == END_THINK:
                    c["end_think_outside_asst"] += 1
                if e == "txt_close":
                    c["txt_close_outside_asst"] += 1
        finish(turn)
    return {"step": int(path.rsplit("-", 1)[1].split(".")[0]), **c}


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex, open(sys.argv[1], "w") as out:
        for res in ex.map(scan, sys.argv[2:]):
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["step"], {k: v for k, v in res.items() if k.startswith("turn_")}, flush=True)
