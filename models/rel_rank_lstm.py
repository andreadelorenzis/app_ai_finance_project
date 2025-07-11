import argparse
import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time

from .data_loading import load_EOD_data, load_relation_data
from .evaluate import evaluate

leaky_relu = lambda x, alpha=0.2: torch.maximum(alpha * x, x)

class ReRaLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rel_encoding_shape, inner_prod=False, flat=False):
        super(ReRaLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inner_prod = inner_prod
        self.flat = flat
        self.rel_weight_layer = nn.Linear(rel_encoding_shape[-1], 1)
        if not inner_prod:
            # Head and tail weight layers for sum weight
            self.head_weight = nn.Linear(input_dim, 1)
            self.tail_weight = nn.Linear(input_dim, 1)
        if flat:
            self.hidden_layer = nn.Linear(input_dim * 2, hidden_dim)
        self.prediction_layer = nn.Linear(
            hidden_dim if flat else input_dim * 2, 1
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, feature, relation, rel_mask):
        batch_size = feature.size(0)
        
        rel_weight = F.leaky_relu(self.rel_weight_layer(relation), 0.2)
        
        if self.inner_prod:
            inner_weight = torch.matmul(feature, feature.transpose(0, 1))
            weight = inner_weight * rel_weight[:, :, -1]
        else:
            head_weight = F.leaky_relu(self.head_weight(feature), 0.2)
            tail_weight = F.leaky_relu(self.tail_weight(feature), 0.2)
            
            all_one = torch.ones(batch_size, 1, device=feature.device)
            weight = (torch.matmul(head_weight, all_one.transpose(0, 1)) + 
                     torch.matmul(all_one, tail_weight.transpose(0, 1)) + 
                     rel_weight[:, :, -1])
        
        weight_masked = F.softmax(rel_mask + weight, dim=0)
        outputs_proped = torch.matmul(weight_masked.t(), feature)
        outputs_concated = torch.cat([feature, outputs_proped], dim=1)
        if self.flat:
            outputs_concated = F.leaky_relu(
                self.hidden_layer(outputs_concated), 0.2
            )
        predicted_return_ratio = self.prediction_layer(outputs_concated)
        return predicted_return_ratio


class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None, 
                 flat=False, gpu=False, in_pro=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        print('#tickers selected:', len(self.tickers))

        raw_price_path = os.path.join(data_path, f'{market_name}_raw_prices.npy')
        self.raw_price_data = np.load(raw_price_path)
        if self.market_name == 'NASDAQ':
            self.raw_price_data = self.raw_price_data[:, :-1]
        print('Raw prices shape:', self.raw_price_data.shape)

        _, self.mask_data, self.gt_data, _ = \
            load_EOD_data(data_path, market_name, self.tickers, self.raw_price_data, steps)
        
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}

        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name])
        )
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)

        self.embedding = np.load(
            os.path.join(self.data_path, '..', 'pretrain', 'pretrain', emb_fname))
        print('embedding shape:', self.embedding.shape)

        assert self.embedding.shape[1] == self.mask_data.shape[1], "Shape mismatch: embedding and mask"
        assert self.embedding.shape[1] == self.gt_data.shape[1], "Shape mismatch: embedding and ground truth"
        assert self.embedding.shape[1] == self.raw_price_data.shape[1], "Shape mismatch: embedding and raw prices"
        
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.embedding.shape[1]
        self.fea_dim = 5

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
        
        return (self.embedding[:, offset, :],
                np.expand_dims(mask_batch, axis=1),
                np.expand_dims(base_price_batch, axis=1),
                np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1))

    def train(self):
        self.model = ReRaLSTMModel(
            input_dim=self.parameters['unit'],
            hidden_dim=self.parameters['unit'],
            rel_encoding_shape=self.rel_encoding.shape,
            inner_prod=self.inner_prod,
            flat=self.flat
        ).to(self.device)

        rel_encoding_tensor = torch.FloatTensor(self.rel_encoding).to(self.device)
        rel_mask_tensor = torch.FloatTensor(self.rel_mask).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.parameters['lr'])

        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_valid_perf = {'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0}
        best_test_perf = {'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0}
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            self.model.train()
            
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            
            for j in range(self.valid_index - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                
                feature = torch.FloatTensor(emb_batch).to(self.device)
                mask = torch.FloatTensor(mask_batch).to(self.device)
                ground_truth = torch.FloatTensor(gt_batch).to(self.device)
                
                optimizer.zero_grad()
                return_ratio = self.model(feature, rel_encoding_tensor, rel_mask_tensor)
                reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
                pre_pw_dif = return_ratio - return_ratio.t()
                gt_pw_dif = ground_truth - ground_truth.t()
                mask_pw = mask @ mask.t()
                rank_loss = torch.mean(F.relu(-(pre_pw_dif * gt_pw_dif) * mask_pw))
                total_loss = reg_loss + self.parameters['alpha'] * rank_loss
                
                total_loss.backward()
                optimizer.step()
                
                tra_loss += total_loss.item()
                tra_reg_loss += reg_loss.item()
                tra_rank_loss += rank_loss.item()
            
            num_batches = self.valid_index - self.parameters['seq'] - self.steps + 1
            print('Train Loss:', tra_loss / num_batches, 
                  tra_reg_loss / num_batches, tra_rank_loss / num_batches)

            # Validation
            self.model.eval()
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index], dtype=float
            )
            
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            
            with torch.no_grad():
                for cur_offset in range(
                    self.valid_index - self.parameters['seq'] - self.steps + 1,
                    self.test_index - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                    
                    feature = torch.FloatTensor(emb_batch).to(self.device)
                    mask = torch.FloatTensor(mask_batch).to(self.device)
                    ground_truth = torch.FloatTensor(gt_batch).to(self.device)
                    
                    return_ratio = self.model(feature, rel_encoding_tensor, rel_mask_tensor)
                    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
                    pre_pw_dif = return_ratio - return_ratio.t()
                    gt_pw_dif = ground_truth - ground_truth.t()
                    mask_pw = mask @ mask.t()
                    rank_loss = torch.mean(F.relu(-(pre_pw_dif * gt_pw_dif) * mask_pw))
                    total_loss = reg_loss + self.parameters['alpha'] * rank_loss
                    
                    val_loss += total_loss.item()
                    val_reg_loss += reg_loss.item()
                    val_rank_loss += rank_loss.item()
                    
                    idx = cur_offset - (self.valid_index - self.parameters['seq'] - self.steps + 1)
                    cur_valid_pred[:, idx] = return_ratio.cpu().numpy()[:, 0]
                    cur_valid_gt[:, idx] = gt_batch[:, 0]
                    cur_valid_mask[:, idx] = mask_batch[:, 0]
            
            val_batches = self.test_index - self.valid_index
            print('Valid MSE:', val_loss / val_batches, 
                  val_reg_loss / val_batches, val_rank_loss / val_batches)
            
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            print('\t Valid performance:', cur_valid_perf)

            # Testing
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index], dtype=float
            )
            
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            
            with torch.no_grad():
                for cur_offset in range(
                    self.test_index - self.parameters['seq'] - self.steps + 1,
                    self.trade_dates - self.parameters['seq'] - self.steps + 1
                ):
                    emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                    
                    feature = torch.FloatTensor(emb_batch).to(self.device)
                    mask = torch.FloatTensor(mask_batch).to(self.device)
                    ground_truth = torch.FloatTensor(gt_batch).to(self.device)
                    
                    return_ratio = self.model(feature, rel_encoding_tensor, rel_mask_tensor)
                    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
                    pre_pw_dif = return_ratio - return_ratio.t()
                    gt_pw_dif = ground_truth - ground_truth.t()
                    mask_pw = mask @ mask.t()
                    rank_loss = torch.mean(F.relu(-(pre_pw_dif * gt_pw_dif) * mask_pw))
                    
                    total_loss = reg_loss + self.parameters['alpha'] * rank_loss
                    
                    test_loss += total_loss.item()
                    test_reg_loss += reg_loss.item()
                    test_rank_loss += rank_loss.item()
                    
                    idx = cur_offset - (self.test_index - self.parameters['seq'] - self.steps + 1)
                    cur_test_pred[:, idx] = return_ratio.cpu().numpy()[:, 0]
                    cur_test_gt[:, idx] = gt_batch[:, 0]
                    cur_test_mask[:, idx] = mask_batch[:, 0]
            
            test_batches = self.trade_dates - self.test_index
            print('Test MSE:', test_loss / test_batches,
                  test_reg_loss / test_batches, test_rank_loss / test_batches)
            
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print('\t Test performance:', cur_test_perf)
            
            if val_loss / val_batches < best_valid_loss:
                best_valid_loss = val_loss / val_batches
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)
                print('Better valid loss:', best_valid_loss)
                # self.save_model(self.model, f'../../data/pretrain/pretrain/{self.market_name}_{self.relation_name}_reralstm_model.pt')
            
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)

        return (best_valid_pred, best_valid_gt, best_valid_mask, best_valid_perf,
                best_test_pred, best_test_gt, best_test_mask, best_test_perf)

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True

    def save_model(self, model, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': self.parameters['unit'],
            'hidden_dim': self.parameters['unit'],
            'rel_encoding_shape': self.rel_encoding.shape,
            'inner_prod': self.inner_prod,
            'flat': self.flat
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = ReRaLSTMModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            rel_encoding_shape=checkpoint['rel_encoding_shape'],
            inner_prod=checkpoint['inner_prod'],
            flat=checkpoint['flat']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        print(f"Model loaded from {path}")
        return self.model

    def predict(self, model, start=None):
        model.eval()
        
        test_pred = np.zeros([len(self.tickers), self.trade_dates - start], dtype=float)
        test_gt = np.zeros_like(test_pred)
        test_mask = np.zeros_like(test_pred)
        
        with torch.no_grad():
            for offset in range(start - self.parameters['seq'] - self.steps + 1,
                                self.trade_dates - self.parameters['seq'] - self.steps + 1):
                emb_batch, mask_batch, _, gt_batch = self.get_batch(offset)
                
                feature = torch.FloatTensor(emb_batch).to(self.device)
                mask = torch.FloatTensor(mask_batch).to(self.device)
                gt = torch.FloatTensor(gt_batch).to(self.device)

                return_ratio = self.model(feature, 
                                  torch.FloatTensor(self.rel_encoding).to(self.device),
                                  torch.FloatTensor(self.rel_mask).to(self.device))

                idx = offset - (start - self.parameters['seq'] - self.steps + 1)
                test_pred[:, idx] = return_ratio.cpu().numpy()[:, 0]
                test_gt[:, idx] = gt_batch[:, 0]
                test_mask[:, idx] = mask_batch[:, 0]
                
            performance = evaluate(test_pred, test_gt, test_mask)
                
            return (
                test_pred,
                test_gt,
                test_mask,
                performance
            )

