import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import math

from .data_loading import load_EOD_data
from .evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.zeros_(param)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        prediction = F.leaky_relu(prediction, 0.2)
        return prediction


class LSTM:
    def __init__(self, data_path, market_name, tickers_fname, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\\t', skip_header=False)
        print('#tickers selected:', len(self.tickers))
        
        raw_price_path = os.path.join(data_path, f'{market_name}_raw_prices.npy')
        self.raw_price_data = np.load(raw_price_path)
        if self.market_name == 'NASDAQ':
            self.raw_price_data = self.raw_price_data[:, :-1]
        print('Raw prices shape:', self.raw_price_data.shape)

        self.eod_data, self.mask_data, self.gt_data, _ = \
            load_EOD_data(data_path, market_name, self.tickers, self.raw_price_data, steps)
        
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.batch_size = len(self.tickers) if batch_size is None else batch_size
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5 # Number of features per day
        self.gpu = gpu

        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        print('device:', self.device)

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)

        base_price_batch = self.raw_price_data[:, offset + seq_len - 1]
        for i in range(len(mask_batch)):
            if base_price_batch[i] < 1e-8:
                mask_batch[i] = 0.0
        
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1)

    def compute_loss(self, pred, ground_truth, mask):
        loss = F.mse_loss(pred * mask, ground_truth * mask)
        return loss

    def train(self):
        model = LSTMModel(self.fea_dim, self.parameters['unit']).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.parameters['lr'])

        best_valid_perf = {'mse': np.inf}
        best_test_perf = {'mse': np.inf}
        best_valid_loss = np.inf

        for epoch in range(self.epochs):
            t1 = time()
            model.train()
            total_loss = 0.0

            batch_offsets = np.arange(0, self.valid_index)
            np.random.shuffle(batch_offsets)

            train_steps = self.valid_index - self.parameters['seq'] - self.steps + 1
            for j in range(train_steps):
                eod_batch, mask_batch, gt_batch = self.get_batch(batch_offsets[j])
                x = torch.tensor(eod_batch, dtype=torch.float32, device=self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
                gt = torch.tensor(gt_batch, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                pred = model(x)
                loss = self.compute_loss(pred, gt, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} | Train MSE: {total_loss / train_steps:.6f}")

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index])
                val_gt = np.zeros_like(val_pred)
                val_mask = np.zeros_like(val_pred)
                val_loss = 0.0

                val_steps = self.test_index - self.parameters['seq'] - self.steps + 1
                for offset in range(self.valid_index - self.parameters['seq'] - self.steps + 1, val_steps):
                    eod_batch, mask_batch, gt_batch = self.get_batch(offset)
                    x = torch.tensor(eod_batch, dtype=torch.float32, device=self.device)
                    mask = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
                    gt = torch.tensor(gt_batch, dtype=torch.float32, device=self.device)

                    pred = model(x)
                    loss = self.compute_loss(pred, gt, mask)
                    val_loss += loss.item()

                    idx = offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)
                    val_pred[:, idx] = pred.squeeze().cpu().numpy()
                    val_gt[:, idx] = gt.squeeze().cpu().numpy()
                    val_mask[:, idx] = mask.squeeze().cpu().numpy()

                avg_val_loss = val_loss / (self.test_index - self.valid_index)
                print(f"Valid MSE: {avg_val_loss:.6f}")
                cur_valid_perf = evaluate(val_pred, val_gt, val_mask)
                print('\tValid performance:', cur_valid_perf)

                if avg_val_loss < best_valid_loss:
                    best_valid_loss = avg_val_loss
                    best_valid_perf = cur_valid_perf
                    print('Better valid loss:', best_valid_loss)
                    # self.save_model(model, f'../../data/pretrain/pretrain/{self.market_name}_lstm_model.pt')
            
            print('Epoch:', epoch, 'Time: %.4f' % (time() - t1))
            
        print('\nBest Valid performance:', best_valid_perf)

    def save_model(self, model, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': self.fea_dim,
            'hidden_units': self.parameters['unit']
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        model = LSTMModel(
            input_dim=checkpoint['input_dim'],
            hidden_units=checkpoint['hidden_units']
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded from {path}")
        return model
    
    def predict(self, model, start=None):
        model.eval()
        with torch.no_grad():
            pred_dim = self.trade_dates - start
            test_pred = np.zeros([len(self.tickers), pred_dim])
            test_gt = np.zeros_like(test_pred)
            test_mask = np.zeros_like(test_pred)
    
            test_end_offset = self.trade_dates - self.parameters['seq'] - self.steps + 1
            for offset in range(start - self.parameters['seq'] - self.steps + 1, test_end_offset):
                eod_batch, mask_batch, gt_batch = self.get_batch(offset)
                
                x = torch.tensor(eod_batch, dtype=torch.float32, device=self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
                gt = torch.tensor(gt_batch, dtype=torch.float32, device=self.device)
                
                prediction = model(x)
                
                idx = offset - (start - self.parameters['seq'] - self.steps + 1)
                test_pred[:, idx] = prediction.squeeze().cpu().numpy()
                test_gt[:, idx] = gt.squeeze().cpu().numpy()
                test_mask[:, idx] = mask.squeeze().cpu().numpy()

            performance = evaluate(test_pred, test_gt, test_mask)
                
            return (test_pred, test_gt, test_mask, performance)