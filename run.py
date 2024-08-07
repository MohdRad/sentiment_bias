import numpy as np
import pandas as pd
from use_model import sentiment_model, bias

cp = ["kumo24/bert-sentiment-nuclear",
      "kumo24/gpt2-sentiment-nuclear",
      "kumo24/falcon-sentiment-nuclear",
      "kumo24/mistralai-sentiment-nuclear"]

llm = ['bert',
       'gpt2',
       'falcon',
       'mistral']

bias_mat = np.zeros((4,1))
for j in range(len(cp)):
    model = sentiment_model(checkpoint=cp[j])
    print(llm[j])
    instances = bias(model,
                    './prompts/energy.csv', 
                    './Results/simple_high.csv')
    bias_mat[j,0] = instances
    
df_bias = pd.DataFrame(bias_mat, 
                           index=['BERT', 'GPT-2', 'Falcon', 'MistralAI'], 
                           columns='energy')
df_bias.to_csv('./Results/energy_inst.csv')