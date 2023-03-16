import os
import shutil
import time
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
model_path = "../../../pretrained_weights/symanto-sn-xlm-roberta-base-snli-mnli-anli-xnli/"
​
class SearchEngine:
    def __init__(self, base_df, model_path, search_columns=[""]):
        self.model = SentenceTransformer(model_path, device="cuda")
        self.search_columns = search_columns
        self.base_ds = self.get_base_ds(base_df) # 因为会新增数据，所以在这里设置一个base_df，如果有新增，则增量添加到该df中。最后存储的话，都是使用Apach Arrow。
        
    def get_embeddings(self, text_list):
        return self.model.encode(text_list)
    
    def get_search_text(self, examples):
        search_text = ""
        for element in self.search_columns:
            search_text += examples[element] + "."
        return {"search_text": search_text[:-1]}
    
    def get_embeddings_list(self, examples):
        return {"embeddings": self.get_embeddings(examples["search_text"])}
    
    def deal_dataset(self, ds):
        ds = ds.map(self.get_search_text, batched=False)
        ds = ds.map(self.get_embeddings_list, batched=True, batch_size=64)
        return ds
    
    def get_base_ds(self, base_df):
        base_ds = Dataset.from_pandas(base_df)
        base_ds = self.deal_dataset(base_ds)
        base_ds.save_to_disk("./base_ds")
        base_ds.add_faiss_index(column="embeddings")
        return base_ds
    
    def find_similar_text(self, text, top_k=5):
        search_embedding = self.model.encode(text)
​
        return self.base_ds.get_nearest_examples(
            "embeddings", search_embedding, k=top_k
        ) # scores, samples
​
    def add_new_dataframe(self, df):
        ds = Dataset.from_pandas(df)
        ds = self.deal_dataset(ds)
        if os.path.exists("./base_ds"):
            base_ds = load_from_disk("./base_ds")
            added_ds = concatenate_datasets([base_ds, ds])
            added_ds.save_to_disk("./base_ds_added")
            added_ds.add_faiss_index(column="embeddings")
            self.base_ds = added_ds
            shutil.rmtree("./base_ds")
        else:
            base_ds = load_from_disk("./base_ds_added")
            added_ds = concatenate_datasets([base_ds, ds])
            added_ds.save_to_disk("./base_ds")
            added_ds.add_faiss_index(column="embeddings")
            self.base_ds = added_ds
            shutil.rmtree("./base_ds_added")