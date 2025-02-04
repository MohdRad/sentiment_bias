This repisotory has the files for a possible publication about social bias quantification in Large Language Models (LLMs). 

# Installation 
The best way to run the codes is using Anaconda. Create an Anaconda environment with Python 3.11 using ```conda create -n NAME python=3.11```, replace NAME with any name, then install the requirments using ```pip install -r requirments.txt```. Sufficient GPU memory is crucial for fine-tuning and testing. Check whether Nvidia-cuda was installed using 

```bash
import torch
print(torch.cuda.is_available())
```
If this prints ```False```, you can download cuda from [Pytorch](https://pytorch.org/get-started/locally/) website.

The models used are: BERT, GPT-2, LLaMA-2-7B, Falcon-7B, and MistralAI-7B. The fine-tuned models with the highest classification accuracy are on my [page](https://huggingface.co/kumo24) on huggingface. **The model and the results for LLaMA-2 are excluded because of the restrictions on its use.** 

For LLaMA-2, you must have a huggingface accout, and you need to grant an access from Meta, you will get it in few hours. If you want to import the model from huggingface, you need to generate a token from your account and use it by huggingface package (```huggingface_hub```). The lines of code below show how to do that in Python. If you want to avoid this last step, download the model and keep it locally on your machine after you get the permission to access the files.

```bash
from huggingface_hub import login
login(token='YOUR TOKEN')
```
These two lines are lines 21 and 22 in ``` ./src/fine_tuning.py```, you can uncomment them and use your token. 

# Fine-tuning and Testing
Fine-tuning of the models using 4,000 general tweets (```./data/zero-shot.csv```) from this Kaggle [dataset](https://www.kaggle.com/datasets/daniel09817/twitter-sentiment-analysis) can be done by running ```python run_fine_tuning.py```. The models fine-tuned using 80\% of this Kaggle dataset (```./data/train_gen.csv''') are availabel on my [page](https://huggingface.co/kumo24) on huggingface. To test the fine-tuned models on a small sample (```./data/sample.csv```), use ``` python run_testing.py```. The classification accuracy is 98\%, you can also test the models using 20\% of the [dataset](https://www.kaggle.com/datasets/daniel09817/twitter-sentiment-analysis) (```./data/test_gen.csv```) but the classification accuracy will also be 98\%. The classification report is written to ```.csv``` file. Both ```run_fine_tuning.py``` & ```run_testing.py``` have comments that describe each input parameter. 


```bash
python run.py
```
