from transformers import AutoTokenizer
import numpy as np
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
import pandas as pd


def sentiment_model (checkpoint):
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    
    
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
        

    if (checkpoint =='kumo24/gpt2-sentiment-nuclear'):
        tokenizer.pad_token = tokenizer.eos_token
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if (checkpoint=="kumo24/bert-sentiment-nuclear"):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                           num_labels=3,
                                                           id2label=id2label, 
                                                           label2id=label2id)
        model.to("cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                           num_labels=3,
                                                           id2label=id2label, 
                                                           label2id=label2id,
                                                           device_map='auto')
  
    if (checkpoint =='kumo24/gpt2-sentiment-nuclear'):
             model.config.pad_token_id = model.config.eos_token_id

    sentiment_task = pipeline("sentiment-analysis", 
                            model=model, 
                            tokenizer=tokenizer)
    return sentiment_task


def bias (model, path, name):
    text = pd.read_csv(path, header=None)
    text = text.values.tolist()
    results = []
    for i in range (len(text)):
        a = model(text[i][0])
        results.append([text[i][0],a[0]['label'],a[0]['score']])
    
    results = np.array(results)    
    biased = []
    l = 0 
    r = 1 
    for i in range (int(len(results)/2)):
        if(results[l,1]==results[r,1]):
            pass
        else:
            biased.append([str(results[l,0]), results[l,1], results[l,2]])
            biased.append([str(results[r,0]), results[r,1], results[r,2]])
        l = l+2
        r = r+2
        
    df = pd.DataFrame(biased,columns=['text','sentiment','score'])
    df.to_csv(name)
    return len(biased)/2


  





