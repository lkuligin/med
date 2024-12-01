import json
import argparse

if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", type=str)

        return vars(parser.parse_args())

    args = parse_args()

    with open(args["input_file"], "r") as json_file:
        data = json.load(json_file)

    questions = data["questions"]

    answers = {"A"}

    for question in questions:
        options = question["options"]
        for option in options:
            answers.add(option["key"])

    print(answers)
