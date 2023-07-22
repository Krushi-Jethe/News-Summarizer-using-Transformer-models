"""

Krushi_Jethe

Though the file has been named abstractive , it
can be used for training both abstractive and extractive
models.

To train models for summarization
After importing , it installs necessary libraries &
Log-in to the huggingface hub

"""



import subprocess

subprocess.run(['pip', 'install', 'transformers', 'datasets', 'evaluate', 'rouge_score'])

import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

from huggingface_hub import notebook_login

notebook_login()

class abstractive_trainer():
    
    """
    
    abstractive_trainer class is used to train a model for abstractive summarization
    
    Usage      :  model = abstractive_trainer(checkpoint , save_to_dir)
                  model.prepare_dataset()
                  model.run(num_epochs)
    
    Class      :  abstractive_trainer
    
    Atrributes :  checkpoint         -- model to be used
                  tokenizer          -- tokenizer to tokenize data
                  prefix             -- tells the model what to do , in this case "summarize" 
                                        (other prefixes could be for Q/A or sentiment analysis)
                  rouge              -- metric for evaluating model
                  data_collator      -- used for bacth processing the data
                  model              -- initializing the model
                  save_to_Dir        -- To save your model to hugging face for deployment
                  tokenized_dataste  -- stores the final dataset to be input to the model
                  
    Methods    :  pre-process function   -- preprocesses the data by tokenizing them
                  prepare_dataset        -- initializes and prepares the dataset
                  compute_metrics        -- Used for evaluating after every epoch
                  run                    -- trains the model
    """
    
    def __init__(self,checkpoint,save_to_dir):
        
        self.checkpoint = checkpoint #"t5-small & distilbart were used while training"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.prefix = "summarize: "
        self.rouge = evaluate.load("rouge")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.save_to_dir = save_to_dir

    def preprocess_function(self,examples):
        
        inputs = [self.prefix + doc for doc in examples["document"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)

        labels = self.tokenizer(text_target=examples["summary"], max_length=250, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
        



    def prepare_dataset(self , name = "multi_news"):  #currently written specifically for this dataset (multinews)
                                                      #can be modified to be used for any dataset and any size(here train,test,val size
                                                      #are 2000,200,200 respectively , modify as per your requirements)

            dataset = load_dataset(name)
            print(dataset)

            # Select a limited number of examples from each subset
            train_datasets = dataset['train'].select(range(2000))
            val_datasets = dataset['validation'].select(range(200))
            test_datasets = dataset['test'].select(range(200))

            # Merge the subsets into a DatasetDict
            dataset = datasets.DatasetDict({
                "train": train_datasets,
                "validation": val_datasets,
                "test": test_datasets
            })

            del train_datasets
            del val_datasets
            del test_datasets

            print(dataset)
            self.tokenized_dataset = dataset.map(self.preprocess_function, batched=True)




    def compute_metrics(self,eval_pred):
        
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}


    def run(self,num_epochs):
        
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.save_to_dir,
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=num_epochs,
                predict_with_generate=True,
                fp16=True,
                push_to_hub=True,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["test"],
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )

            trainer.train()
