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

class RankLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(RankLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Shape: [batch, seq_len, hidden]
        last_output = lstm_out[:, -1, :]  # Last time step
        raw_pred = self.fc(last_output)
        pred = self.leaky_relu(raw_pred)
        return pred


class RankLSTM:
    def __init__(self, data_path, market_name, tickers_fname, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        print('#tickers selected:', len(self.tickers))
        
        # Load the RAW prices for loss calculation
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
        self.fea_dim = 5
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
               np.expand_dims(base_price_batch, axis=1), \
               np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1)

    def compute_losses(self, pred, base_price, ground_truth, mask, alpha):
        return_ratio = (pred - base_price) / base_price
    
        reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    
        pre_pw_dif = return_ratio - return_ratio.t()
        gt_pw_dif = ground_truth - ground_truth.t()
        mask_pw = mask @ mask.t()
        rank_loss = torch.mean(F.relu(-(pre_pw_dif * gt_pw_dif) * mask_pw))
    
        total_loss = reg_loss + alpha * rank_loss
        return total_loss, reg_loss, rank_loss, return_ratio

    def train(self):
        model = RankLSTMModel(self.fea_dim, self.parameters['unit']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.parameters['lr'])

        best_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_valid_gt = np.zeros_like(best_valid_pred)
        best_valid_mask = np.zeros_like(best_valid_pred)

        best_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.parameters['seq'] -
                                   self.test_index - self.steps + 1], dtype=float)
        best_test_gt = np.zeros_like(best_test_pred)
        best_test_mask = np.zeros_like(best_test_pred)

        best_valid_perf = {'mse': np.inf}
        best_test_perf = {'mse': np.inf}
        best_valid_loss = np.inf

        for epoch in range(self.epochs):
            t1 = time()
            model.train()
            total_loss, total_reg_loss, total_rank_loss = 0.0, 0.0, 0.0

            batch_offsets = np.arange(0, self.valid_index)
            np.random.shuffle(batch_offsets)

            for j in range(self.valid_index - self.parameters['seq'] - self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                x = torch.tensor(eod_batch, dtype=torch.float32, device=device)
                mask = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                base_price = torch.tensor(price_batch, dtype=torch.float32, device=device)
                gt = torch.tensor(gt_batch, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                pred = model(x)
                loss, reg_loss, rank_loss, _ = self.compute_losses(pred, base_price, gt, mask, self.parameters['alpha'])
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_reg_loss += reg_loss.item()
                total_rank_loss += rank_loss.item()

            denom = self.valid_index - self.parameters['seq'] - self.steps + 1
            print('Train Loss:',
                  total_loss / denom,
                  total_reg_loss / denom,
                  total_rank_loss / denom)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
                val_gt = np.zeros_like(val_pred)
                val_mask = np.zeros_like(val_pred)
                val_loss = val_reg_loss = val_rank_loss = 0.0

                for offset in range(self.valid_index - self.parameters['seq'] - self.steps + 1,
                                    self.test_index - self.parameters['seq'] - self.steps + 1):
                    eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(offset)
                    x = torch.tensor(eod_batch, dtype=torch.float32, device=device)
                    mask = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                    base_price = torch.tensor(price_batch, dtype=torch.float32, device=device)
                    gt = torch.tensor(gt_batch, dtype=torch.float32, device=device)

                    pred = model(x)
                    loss, reg_loss, rank_loss, return_ratio = self.compute_losses(pred, base_price, gt, mask,
                                                                             self.parameters['alpha'])

                    val_loss += loss.item()
                    val_reg_loss += reg_loss.item()
                    val_rank_loss += rank_loss.item()
                    idx = offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)
                    val_pred[:, idx] = return_ratio.squeeze().cpu().numpy()
                    val_gt[:, idx] = gt.squeeze().cpu().numpy()
                    val_mask[:, idx] = mask.squeeze().cpu().numpy()

                denom = self.test_index - self.valid_index
                print('Valid MSE:',
                      val_loss / denom,
                      val_reg_loss / denom,
                      val_rank_loss / denom)
                cur_valid_perf = evaluate(val_pred, val_gt, val_mask)
                print('\tValid performance:', cur_valid_perf)

                # Testing
                test_pred = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
                test_gt = np.zeros_like(test_pred)
                test_mask = np.zeros_like(test_pred)
                test_loss = test_reg_loss = test_rank_loss = 0.0

                for offset in range(self.test_index - self.parameters['seq'] - self.steps + 1,
                                    self.trade_dates - self.parameters['seq'] - self.steps + 1):
                    eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(offset)
                    x = torch.tensor(eod_batch, dtype=torch.float32, device=device)
                    mask = torch.tensor(mask_batch, dtype=torch.float32, device=device)
                    base_price = torch.tensor(price_batch, dtype=torch.float32, device=device)
                    gt = torch.tensor(gt_batch, dtype=torch.float32, device=device)

                    pred = model(x)
                    loss, reg_loss, rank_loss, return_ratio = self.compute_losses(pred, base_price, gt, mask,
                                                                             self.parameters['alpha'])

                    test_loss += loss.item()
                    test_reg_loss += reg_loss.item()
                    test_rank_loss += rank_loss.item()
                    idx = offset - (self.test_index - self.parameters['seq'] - self.steps + 1)
                    test_pred[:, idx] = return_ratio.squeeze().cpu().numpy()
                    test_gt[:, idx] = gt.squeeze().cpu().numpy()
                    test_mask[:, idx] = mask.squeeze().cpu().numpy()

                denom = self.trade_dates - self.test_index
                print('Test MSE:',
                      test_loss / denom,
                      test_reg_loss / denom,
                      test_rank_loss / denom)
                cur_test_perf = evaluate(test_pred, test_gt, test_mask)
                print('\tTest performance:', cur_test_perf)

                if val_loss / (self.test_index - self.valid_index) < best_valid_loss:
                    best_valid_loss = val_loss / (self.test_index - self.valid_index)
                    best_valid_perf = copy.deepcopy(cur_valid_perf)
                    best_valid_pred = val_pred.copy()
                    best_valid_gt = val_gt.copy()
                    best_valid_mask = val_mask.copy()
                    best_test_perf = copy.deepcopy(cur_test_perf)
                    best_test_pred = test_pred.copy()
                    best_test_gt = test_gt.copy()
                    best_test_mask = test_mask.copy()
                    print('Better valid loss:', best_valid_loss)
                    self.save_model(model, f'../../data/pretrain/pretrain/{self.market_name}_ranklstm_model.pt')

            print('Epoch:', epoch, 'Time: %.4f' % (time() - t1))

        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)

        return best_valid_pred, best_valid_gt, best_valid_mask, best_valid_perf, \
               best_test_pred, best_test_gt, best_test_mask, best_test_perf

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True

    def save_model(self, model, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': self.fea_dim,
            'hidden_units': self.parameters['unit']
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        model = RankLSTMModel(
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
            test_pred = np.zeros([len(self.tickers), self.trade_dates - start], dtype=float)
            test_gt = np.zeros_like(test_pred)
            test_mask = np.zeros_like(test_pred)
    
            for offset in range(start - self.parameters['seq'] - self.steps + 1,
                                self.trade_dates - self.parameters['seq'] - self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(offset)
                x = torch.tensor(eod_batch, dtype=torch.float32, device=self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
                base_price = torch.tensor(price_batch, dtype=torch.float32, device=self.device)
                gt = torch.tensor(gt_batch, dtype=torch.float32, device=self.device)
                pred = model(x)
                loss, reg_loss, rank_loss, return_ratio = self.compute_losses(pred, base_price, gt, mask,
                                                                             self.parameters['alpha'])
                idx = offset - (start - self.parameters['seq'] - self.steps + 1)
                test_pred[:, idx] = return_ratio.squeeze().cpu().numpy()
                test_gt[:, idx] = gt.squeeze().cpu().numpy()
                test_mask[:, idx] = mask.squeeze().cpu().numpy()

            performance = evaluate(test_pred, test_gt, test_mask)
                
            return (
                test_pred,
                test_gt,
                test_mask,
                performance
            )