# med

`jupyter notebook --NotebookApp.token="" --NotebookApp.allow_origin="*" --NotebookApp.open_browser=False`

```
cd med/med
export PYTHONPATH=$PYTHONPATH:.
python qa/run.py --model_name=gemini-1.5-pro-001 \
 --sample_size=10 --output_file_name=experiment2a.json

python qa/run.py --model_name=gemini-1.0-pro-001 \
 --sample_size=10 --output_file_name=experiment1a.json
```