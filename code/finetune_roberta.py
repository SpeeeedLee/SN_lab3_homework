import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

#############################################################
# Use "Pytorch Lightning" Framework to finetune the model 
# The classifier head is build with :
#   One Hidden Layer
#   One Classify Layer

# All parameters in the model are trainable, not just the head
# Some dropout is used to alleviate the data imbalace issue                    
###############################################################    



def preprocess_tweet(tweet):
    '''
    Replace all user tags to "@user"
    Replace all web link to  "http"
    Remove meaningless word : "<LH>" 

    Since the LLM model I used is already trained with lots of tweets, 
        so instead of removing the user tags and web link, I "Mark" them. 
    '''
    tweet_words = []
    for word in tweet.split(' '):

        if word.startswith('@') and len(word) > 1:
            word = '@user'

        elif word.startswith('http'):
            word = "http"

        word = word.replace('<LH>', '').replace('<lh>', '').replace('<', '').replace('>', '')
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    return tweet_proc


class EmotionDataset(Dataset):
    '''
    Dataset class for training and validation
    '''
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
class EmotionTestDataset(Dataset):
    '''
    Dataset class for testing
    '''
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }





class Emotion_Classifier(pl.LightningModule):
    '''
    The main model class
    '''

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = 0
        if labels is not None:
            loss = self.loss_func(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        _, logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        return preds


    def configure_optimizers(self):
        '''
        I use the cosine warmup scheduler
        '''
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size'] / self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)   # cosine warmup scheduler
        return [optimizer], [scheduler]
    

if __name__ == '__main__':

    logger = TensorBoardLogger("tensorboard_logs")               # Use tensor_board to record Epoch_num, Training_loss
    train_df = pd.read_csv('../datasets/train_data_clean.csv')
    test_df = pd.read_csv('../datasets/test_data_clean.csv')

    # Shuffle the training dataset
    train_df = train_df.sample(frac=1.0, random_state=42)

    # Create Dict for label <--> ID
    unique_labels = train_df['emotion'].unique()
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()} 

    # Draw "text" and "emotion" out, then convert to list
    train_texts = train_df['text'].tolist()
    train_labels = train_df['emotion'].apply(lambda x: label_to_id[x]).tolist() 
    test_texts = test_df['text'].tolist()

    # Preprocess
    train_texts_processed = [preprocess_tweet(tweet) for tweet in train_texts]
    test_texts_processed = [preprocess_tweet(tweet) for tweet in test_texts]

    # Use the model that already train on Tweets
    model_name = 'cardiffnlp/twitter-roberta-base-2022-154m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 64     # See maximum 64 words in one tweet !

    # Create dataset instance
    train_dataset = EmotionDataset(train_texts_processed, train_labels, tokenizer, max_length)
    test_dataset = EmotionTestDataset(test_texts_processed, tokenizer, max_length)
    
    # Create dataset loader instance
    train_loader = DataLoader(train_dataset, batch_size = 980, shuffle=True, num_workers=47)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=47)


    # Configuration of the model
    config = {
        'model_name': 'cardiffnlp/twitter-roberta-base-2022-154m', 
        'n_labels': len(unique_labels), 
        'lr': 3e-4, 
        'weight_decay': 0.01,  
        'batch_size': 980,  
        'train_size': len(train_dataset),
        'warmup': 0.15  
    }

    # Instantiate the model
    model = Emotion_Classifier(config)


    trainer = Trainer(
        logger=logger,
        log_every_n_steps = 1,
        max_epochs=12,  
        #progress_bar_refresh_rate=30,
        #val_check_interval=1000,   # Only used when still use valildation set to find the hyperparameter
        accelerator="gpu", 
        devices=[1]
    )


    ################### Train the model #####################
    trainer.fit(model, train_loader)

    ################### Predict the Results #####################
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    predictions = trainer.predict(model, dataloaders=test_loader)
    predictions = [pred for batch in predictions for pred in batch.cpu().numpy()]
    emotion_predictions = [id_to_label[pred] for pred in predictions]

    # Create Final DataFrame
    final_test_df = pd.DataFrame({
        "id": test_df["tweet_id"],
        "emotion": emotion_predictions
    })

    # Save to csv file
    final_test_df.to_csv('../datasets/predict_final.csv', index=False)

    print("csv file 儲存成功!!!!")
