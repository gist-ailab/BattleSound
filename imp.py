import numpy as np


if __name__=='__main__':
    with open('result_ablation.txt', 'r') as f:
        result = f.readlines()
        
    acc_list = []
    
    for string in result:
        ind, acc = string.strip().split('_')
        ind = int(ind)
        acc = float(acc)
        if ind % 5 == 0:
            acc_list = np.array(acc_list)
            print(ind-1, '--', np.mean(acc_list), '--', np.std(acc_list))
            acc_list = []
        
        acc_list.append(acc)
    
    print(ind-1, '--', np.mean(acc_list), '--', np.std(acc_list))
    
        
            
        
        
        
