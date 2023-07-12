# LLM Tuner

Generic trainer script to fine tune LLMs in an alpaca formatted dataset.

The trainer will use the text data in the `"text"` column to tune the LLM.

Check more about the alpaca format [here](https://huggingface.co/datasets/tatsu-lab/alpaca)

### Instructions:
- Install requirements with `pip install -r requirements.txt`
- Customize `train.py` script training variables according to your training specs
- Run `train.py`

### Credits:
This is based on the work of Abhishek Thakur in the video [Train LLMs in just 50 lines of code!](https://www.youtube.com/watch?v=JNMVulH7fCo)

### TODOs:
- Add modifiable command line arguments
