"""Facts judged after the stop at a candidate's first wrong one.

The verdicts already stored stay as they are; the missing ones are appended in
the facts' order, so a verdict's position is still its fact's.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gdanschin_runtime.complete_verdicts import complete, cut_short  # noqa: E402


def verdict(value, **extra):
    return {"fact": "f", "verdict": value, "is_correct": bool(value), **extra}


def test_only_a_stop_at_a_wrong_fact_before_the_last_counts_as_cut_short():
    assert cut_short({"facts": ["a", "b", "c"], "fact_verifications": [verdict(1), verdict(0)]})
    assert not cut_short({"facts": ["a", "b"], "fact_verifications": [verdict(1), verdict(0)]})
    assert not cut_short({"facts": ["a", "b", "c"], "fact_verifications": [verdict(1)]})
    assert not cut_short({"facts": ["a", "b", "c"], "fact_verifications": []})
    # judged out of order, cited facts first: positions are not the facts'
    assert not cut_short({"facts": ["a", "b", "c"],
                          "fact_verifications": [verdict(1, fact_index=2), verdict(0, fact_index=0)]})


class FakeJudge:
    def __init__(self):
        self.asked = []

    async def verify_single_fact(self, question_text, options, fact, session_id, user_id, semaphore):
        self.asked.append(fact)
        tokens = SimpleNamespace(total_tokens=10, prompt_tokens=7, candidate_tokens=3)
        return SimpleNamespace(tokens=tokens, to_dict=lambda: verdict(1, fact=fact))


def test_the_unchecked_facts_are_judged_and_appended_in_order(tmp_path):
    judge_dir = tmp_path / "question_0" / "glm"
    judge_dir.mkdir(parents=True)
    (judge_dir / "result.json").write_text(json.dumps({"question": "Q", "options": {"A": "a"}}))
    path = judge_dir / "iteration_3.json"
    path.write_text(json.dumps({"verification": {
        "candidate_index": 3, "facts": ["a", "b", "c", "d"], "all_facts_correct": False,
        "fact_verifications": [verdict(1, fact="a"), verdict(0, fact="b")], "total_tokens": 20}}))
    judge = FakeJudge()
    added = asyncio.run(complete(judge, path, asyncio.Semaphore(4)))
    v = json.loads(path.read_text())["verification"]
    assert (added, judge.asked) == (2, ["c", "d"])
    assert [f["fact"] for f in v["fact_verifications"]] == ["a", "b", "c", "d"]
    assert [f["verdict"] for f in v["fact_verifications"]] == [1, 0, 1, 1]
    assert v["all_facts_correct"] is False
    assert (v["total_tokens"], v["facts_judged_after_stop"]) == (40, 2)
    assert not cut_short(v)
