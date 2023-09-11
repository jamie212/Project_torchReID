import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configure the thresholds')
    parser.add_argument('--similarity_th', type=float, default=0.8, help='相似度達閥值當作同一物品')
    parser.add_argument('--high_iou_th', type=float, default=0.98, help='iou達此閥值的話即使相似度低也沒關係')
    parser.add_argument('--high_iou_simi_th', type=float, default=0.7, help='iou高時相似度閥值可較低')
    parser.add_argument('--stay_up_th', type=int, default=400, help='存在超過此幀數，有可能是垃圾')
    parser.add_argument('--overlap_th', type=float, default=0.3, help='iou沒達到此閥值不會是同物體')
    parser.add_argument('--delete_cnt', type=int, default=5, help='連續幾幀沒被配到前景就會被刪掉')
    parser.add_argument('--covered_th', type=float, default=0.8, help='被覆蓋的面積佔自己的比例超過此閥值認定為被覆蓋')
    parser.add_argument('--moved_th', type=int, default=200, help='累積移動量超過此閥值不會是垃圾')
    parser.add_argument('--be_covered_time_th', type=int, default=300, help='累積被覆蓋了幀數超過此閥值，會被更新/刪掉')
    parser.add_argument('--reset_cnt', type=int, default=30, help='le of tracking list達到此數字，清空')


    return parser.parse_args()
