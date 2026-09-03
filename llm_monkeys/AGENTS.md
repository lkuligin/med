This is the implementation of the inference scaling approach in medical domain inspired by the [paper](https://arxiv.org/abs/2407.21787) "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling". It uses the public MedQA dataset (it's TEST part).

The proposed research flow:
1. Answer questions multiple times (n=3) with different SLMs (Gemma 3, Gemma 4) and split the MedQA questions into "simple" and "difficult" ones. The simple ones are answered correctly by all attempts by any LLM (not being grounded and using internal knowledge only).
2. Generate N=1000 candidates using an SLM (passed as a configuration) for each complex question. Each candidate generation should be two steps:
 - generating medical facts (as a structured output). Each fact is an atomic verifiable statement that is relevant to answer the question
 - generate the final answer based on the facts generated
Candidates should be written into a json file, and metadata like tokens and execution time should be collected.
3. Run a verification of each fact with a separate LLM. Once all facts are marked as correct for a given candidate, treat them as a final answer.


Key requirements for the code:
1. Use ADK for agent development.
2. When consuming LLM from Vertex, use ADK lite-llm adapter if possible.
