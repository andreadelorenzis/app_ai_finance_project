import os
import numpy as np

def load_EOD_data(data_path, market_name, tickers, raw_price_matrix, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = [] 
    
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            single_EOD = single_EOD[:-1, :]

        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0], single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)

        ticker_raw_prices = raw_price_matrix[index]

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            
            elif row >= steps:
                price_today = ticker_raw_prices[row]
                price_yesterday = ticker_raw_prices[row - steps]

                if price_yesterday > 1e-8 and price_today > -1000 and price_yesterday > -1000:
                    ground_truth[index][row] = (price_today - price_yesterday) / price_yesterday
                else:
                    ground_truth[index][row] = 0.0
                    masks[index][row] = 0.0

            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1

        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]

    return eod_data, masks, ground_truth, base_price


def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape) 
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]] 
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int), np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask
