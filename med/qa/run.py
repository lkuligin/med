import argparse
import json
import time
from collections import Counter

from grpc._channel import _InactiveRpcError

from qa.chains import get_chain
from qa.callbacks import get_callback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--chain_type", type=str, default="simple")
    parser.add_argument("--sample_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--input_file", type=str, default="med_qa_train.json")
    parser.add_argument("--callback", action="store_true")

    return vars(parser.parse_args())


class Sampler:
    def __init__(self, chain, max_retries: int = 10):
        self._chain = chain
        self._max_retries = max_retries

    def run(self, entry, n=10, stream: bool = False, *, callback = None):
        answers = [self._run(entry, stream=stream, callback=callback) for _ in range(n)]
        results = [a["answer"] for a in answers]
        counter = Counter(results)
        winner, _ = counter.most_common()[0]
        return winner

    def _run(self, entry, retry: int = 0, stream: bool = False, *, callback = None):
        if retry > self._max_retries:
            raise ValueError("Max retries reached")
        try:
            kwargs = {}
            if callback:
                kwargs = {"config": {"callbacks": [callback]}}
            if stream:
                res = list(self._chain.stream(entry, **kwargs))
                return {k: v for c in res for k, v in c.items()}
            else:
                return self._chain.invoke(entry, **kwargs)
        except _InactiveRpcError:
            time.sleep(1)
            return self._run(entry, retry + 1)
        except Exception:
            time.sleep(1)
            return self._run(entry, retry + 1)


def run():
    args = parse_args()
    print(args)
    if args["chain_type"] in []:
        pass
    else:
        chain = get_chain(
            sample_size=args["sample_size"],
            chain_type=args["chain_type"],
            model_name=args["model_name"],
            temperature=args["temperature"],
        )
        sampler = Sampler(chain=chain)
    results = []

    with open(args["input_file"], "r") as json_file:
        data = json.load(json_file)
    use_callback = args.get("callback", False)

    results = []
    start = 0
    try:
        with open(args["output_file_name"], "r") as json_file:
            results = json.load(json_file)["results"]
            start = len(results)
        print(f"Loaded previous run, starting from pos {start}.")
    except FileNotFoundError:
        pass

    for i, entry in enumerate(data["questions"][start:]):
        answer_idx = entry["answer_idx"]
        # stream = True if args["model_name"] in ["llama_3_405b"] else False
        stream = False
        callback = get_callback(model_name=args["model_name"]) if use_callback else None
        start_time = time.time()
        result = sampler.run(entry, n=args["sample_size"], stream=stream, callback=callback)
        result = {
                "correct_answer_idx": answer_idx,
                "genai_answer_idx": result,
                "latency": time.time() - start_time
        }
        if callback:
            result = {**result, **{
                "prompt_tokens": callback.prompt_tokens,
                "completion_tokens": callback.completion_tokens,
                "max_input_tokens": callback.max_input_tokens
            }
        }
        results.append(result)

        if i % 5 == 0:
            print(f"Processed {i} entries")
            with open(args["output_file_name"], "w") as json_file:
                json.dump({"results": results}, json_file, indent=4)

    with open(args["output_file_name"], "w") as json_file:
        json.dump({"results": results}, json_file, indent=4)


if __name__ == "__main__":
    for _ in range(10):
        try:
            run()
        except ValueError as e:
            print(e)
            pass
