import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_name", type=str)

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    with open(args["output_file_name"], "r") as json_file:
        print(args["output_file_name"])
        results = json.load(json_file)["results"]
        questions_count = len(results)
        print(f"Questions: {questions_count}")
        correct_count = 0
        for q in results:
            if q["genai_answer_idx"].strip() == q["correct_answer_idx"].strip():
                correct_count += 1
        print(f"Correct: {correct_count}")
        rate = round(correct_count * 100 / questions_count, 2)
        print(f"Success: {rate}%")
        input_avg_length = round(
            sum([q["prompt_tokens"] for q in results]) / questions_count, 2
        )
        print(f"Input avg token count {input_avg_length}")
        output_avg_length = round(
            sum([q["completion_tokens"] for q in results]) / questions_count, 2
        )
        print(f"Output avg token count {output_avg_length}")
        latency_avg = round(sum([q["latency"] for q in results]) / questions_count, 2)
        print(f"Latency avg {latency_avg}")
