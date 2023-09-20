import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
from sentence_transformers import SentenceTransformer



df = pd.read_csv("data/train.csv", sep = ";")

def compile_text(x):


    text =  f"""Age: {x['age']},  
                housing load: {x['housing']}, 
                Job: {x['job']}, 
                Marital: {x['marital']}, 
                Education: {x['education']}, 
                Default: {x['default']}, 
                Balance: {x['balance']}, 
                Personal loan: {x['loan']}
            """

    return text

sentences = df.apply(lambda x: compile_text(x), axis=1).tolist()



model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

output = model.encode(sentences=sentences, show_progress_bar= True, normalize_embeddings  = True)

df_embedding = pd.DataFrame(output)
df_embedding


df_embedding.to_csv("embedding_train.csv",index = False)
