import argparse
import time
import json
from grpc._channel import _InactiveRpcError

from qa.chains import get_chain
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--chain_type", type=str, default="simple")
    parser.add_argument("--sample_size", type=int, default=1)

    return vars(parser.parse_args())


class Sampler:
    def __init__(self, chain, max_retries: int = 10):
        self._chain = chain
        self._max_retries = max_retries

    def run(self, entry, n=10):
        answers = [self._run(entry) for _ in range(n)]
        results = [a["answer"] for a in answers]
        counter = Counter(results)
        winner, _ = counter.most_common()[0]
        return winner

    def _run(self, entry, retry: int = 0):
        if retry > self._max_retries:
            raise ValueError("Max retries reached")
        try:
            return self._chain.invoke(entry)
        except _InactiveRpcError:
            time.sleep(1)
            return self._run(entry, retry + 1)
        except Exception as e:
            time.sleep(1)
            return self._run(entry, retry + 1)


def run():
    args = parse_args()
    print(args)
    chain = get_chain(
        sample_size=args["sample_size"],
        chain_type=args["chain_type"],
        model_name=args["model_name"],
    )
    results = []

    with open("med_qa_train.json", "r") as json_file:
        data = json.load(json_file)

    sampler = Sampler(chain=chain)

    for i, entry in enumerate(data["questions"]):
        answer_idx = entry["answer_idx"]
        result = sampler.run(entry, n=args["sample_size"])
        results.append(
            {
                "correct_answer_idx": answer_idx,
                "genai_answer_idx": result,
            }
        )

        if i % 50 == 0:
            print(f"Processed {i} entries")
            with open(args["output_file_name"], "w") as json_file:
                json.dump({"results": results}, json_file, indent=4)

    with open(args["output_file_name"], "w") as json_file:
        json.dump({"results": results}, json_file, indent=4)


if __name__ == "__main__":
    run()
