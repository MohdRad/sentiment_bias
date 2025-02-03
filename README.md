This repisotory has the files for a possible publication about social bias quantification in Large Language Models (LLMs). The models used are: BERT, GPT-2, LLaMA-2-7B, Falcon-7B, and MistralAI-7B. The fine-tuned models with the highest classification accuracy are on my page on huggingface https://huggingface.co/kumo24. The results for LLaMA-2 is excluded because of the restrictions on its use. 

For LLaMA-2, you must have a huggingface accout, and you need to grant an access from Meta, you will get it in few hours. If you want to import the model from huggingface, you need to generate a token from your account and use it by huggingface package (```bash huggingface_hub```). The line of codes below shows how to do that in Python. If you want to avoid this last step, download the model and keep it locally on your machine after you get the permission to access the files.

```bash
from huggingface_hub import login
login(token='YOUR TOKEN')
```
The code works on Python 3.11 and later versions, check requirements.txt for the required packages.


```bash
python run.py
```
