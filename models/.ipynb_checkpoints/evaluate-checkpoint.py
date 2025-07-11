import numpy as np
import matplotlib.pyplot as plt

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2 / np.sum(mask)

    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    for i in range(prediction.shape[1] - 1):
        
        day_gt = ground_truth[:, i + 1]
        day_mask = mask[:, i + 1]
        
        day_pred = prediction[:, i]
        
        valid_indices = np.where(day_mask > 0.5)[0]
        if len(valid_indices) == 0:
            all_miss_days_top += 1
            continue

        rank_gt_valid = np.argsort(day_gt[valid_indices])[::-1]
        rank_pre_valid = np.argsort(day_pred[valid_indices])[::-1]

        pre_top1_idx = valid_indices[rank_pre_valid[:1]]
        pre_top5_idx = valid_indices[rank_pre_valid[:5]]
        pre_top10_idx = valid_indices[rank_pre_valid[:10]]
        
        gt_top1_ticker = valid_indices[rank_gt_valid[0]]
        rank_list = valid_indices[rank_pre_valid]
        position_list = np.where(rank_list == gt_top1_ticker)[0]
        if len(position_list) > 0:
            mrr_top += 1.0 / (position_list[0] + 1)
        else:
            all_miss_days_top +=1


        bt_long += np.mean(day_gt[pre_top1_idx])
        bt_long5 += np.mean(day_gt[pre_top5_idx])
        bt_long10 += np.mean(day_gt[pre_top10_idx])

    effective_days = (prediction.shape[1] -1) - all_miss_days_top
    performance['mrrt'] = mrr_top / effective_days if effective_days > 0 else 0
    
    performance['btl'] = bt_long
    performance['btl5'] = bt_long5
    performance['btl10'] = bt_long10
    
    return performance