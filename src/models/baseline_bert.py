# src/models/baseline_bert.py

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from config import MODEL_NAME, DEVICE, MAX_LENGTH, LEARNING_RATE

class BaselineBertClassifier:
    def __init__(self, num_classes):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_classes, output_attentions=False, output_hidden_states=False)
        self.device = torch.device(DEVICE)
        self.model.to(self.device)

    def _prepare_data_for_loader(self, texts, labels):
        encoded_inputs = self.tokenizer(
            texts, add_special_tokens=True, max_length=MAX_LENGTH,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        labels_tensor = torch.tensor(labels)
        return TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], labels_tensor)

    def train(self, train_texts, train_labels, epochs=1, batch_size=16):
        train_dataset = self._prepare_data_for_loader(train_texts, train_labels)
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        self.model.train()
        for epoch_i in range(epochs):
            print(f'======== Baseline Training: Epoch {epoch_i + 1} / {epochs} ========')
            for batch in train_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(self.device) for t in batch]
                self.model.zero_grad()
                result = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = result.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

    def predict_proba(self, texts, batch_size=32):
        self.model.eval()
        dummy_labels = [0] * len(texts)
        pred_dataset = self._prepare_data_for_loader(texts, dummy_labels)
        pred_dataloader = DataLoader(pred_dataset, sampler=SequentialSampler(pred_dataset), batch_size=batch_size)
        logits = []
        with torch.no_grad():
            for batch in pred_dataloader:
                b_input_ids, b_input_mask, _ = [t.to(self.device) for t in batch]
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits.append(outputs.logits.cpu())
        probabilities = torch.nn.functional.softmax(torch.cat(logits, dim=0), dim=1)
        return probabilities.numpy()