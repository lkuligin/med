# med

`jupyter notebook --NotebookApp.token="" --NotebookApp.allow_origin="*" --NotebookApp.open_browser=False`

```
cd med && export PYTHONPATH=$PYTHONPATH:.
python qa/run.py --model_name=gemini-1.5-pro-001 \
 --sample_size=10 --output_file_name=experiment2a.json

python qa/run.py --model_name=gemini-1.0-pro-001 \
 --sample_size=10 --output_file_name=experiment1a.json

python qa/run.py --model_name=gemini-1.0-pro-001  --sample_size=1 --output_file_name=experiment1.json
python qa/run.py --model_name=gemini-1.0-pro-001  --sample_size=10 --output_file_name=experiment1a.json --temperature=0.5
python qa/run.py --model_name=gemini-1.0-pro-001  --sample_size=1 --output_file_name=experiment1b.json --chain_type=cot
python qa/run.py --model_name=gemini-1.5-pro-001  --sample_size=1 --output_file_name=experiment2.json
python qa/run.py --model_name=gemini-1.5-pro-001  --sample_size=10 --output_file_name=experiment2a.json --temperature=0.5
python qa/run.py --model_name=gemini-1.5-pro-001  --sample_size=1 --output_file_name=experiment2b.json --chain_type=cot
python qa/run.py --model_name=gemini-1.0-pro-002  --sample_size=1 --output_file_name=experiment3.json
python qa/run.py --model_name=gemini-1.0-pro-002  --sample_size=10 --output_file_name=experiment3a.json --temperature=0.5
python qa/run.py --model_name=gemini-1.0-pro-002  --sample_size=1 --output_file_name=experiment3b.json --chain_type=cot
python qa/run.py --model_name=gemini-1.5-flash-001  --sample_size=1 --output_file_name=experiment4.json 
python qa/run.py --model_name=gemini-1.5-flash-001  --sample_size=10 --output_file_name=experiment4a.json --temperature=0.5
python qa/run.py --model_name=gemini-1.5-flash-001  --sample_size=1 --output_file_name=experiment4b.json --chain_type=cot

python qa/run.py --model_name=llama_3_405b  --sample_size=1 --output_file_name=experiment5.json
python qa/run.py --model_name=llama_3_405b  --sample_size=10 --output_file_name=experiment5a.json --temperature=0.5
python qa/run.py --model_name=llama_3_405b  --sample_size=1 --output_file_name=experiment5b.json --chain_type=cot

python qa/run.py --model_name=llama_2b  --sample_size=1 --output_file_name=experiment6.json
python qa/run.py --model_name=llama_2b  --sample_size=10 --output_file_name=experiment6a.json --temperature=0.5
python qa/run.py --model_name=llama_2b  --sample_size=1 --output_file_name=experiment6b.json --chain_type=cot --max_output_tokens=80


python qa/run.py --model_name=anthropic_claude  --sample_size=1 --output_file_name=experiment7.json
python qa/run.py --model_name=anthropic_claude  --sample_size=10 --output_file_name=experiment7a.json --temperature=0.5
python qa/run.py --model_name=anthropic_claude  --sample_size=1 --output_file_name=experiment7b.json --chain_type=cot

python qa/run.py --model_name=gemma_2b  --sample_size=1 --output_file_name=experiment8.json
python qa/run.py --model_name=gemma_2b  --sample_size=10 --output_file_name=experiment8a.json --temperature=0.5
python qa/run.py --model_name=gemma_2b  --sample_size=1 --output_file_name=experiment8b.json --chain_type=cot --max_output_tokens=150


python qa/run.py --model_name=mistral_large  --sample_size=1 --output_file_name=experiment9.json
python qa/run.py --model_name=mistral_large  --sample_size=10 --output_file_name=experiment9a.json --temperature=0.5
python qa/run.py --model_name=mistral_large  --sample_size=1 --output_file_name=experiment9b.json --chain_type=cot 

python qa/run.py --model_name=mistral_nemo  --sample_size=1 --output_file_name=experiment10.json
python qa/run.py --model_name=mistral_nemo  --sample_size=10 --output_file_name=experiment10a.json --temperature=0.5
python qa/run.py --model_name=mistral_nemo  --sample_size=10 --output_file_name=experiment10b.json --chain_type=cot


python qa/run.py --model_name=gemma_2b_it  --sample_size=1 --output_file_name=experiment11.json


```