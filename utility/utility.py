import pickle as pkl

def transposer(mean_ate_, mse_ate_, bias_ate_, var_ate_, test_sizes, k):

    a = mean_ate_.reshape(len(test_sizes),k).transpose()
    b = mse_ate_.reshape(len(test_sizes),k).transpose()
    c = bias_ate_.reshape(len(test_sizes),k).transpose()
    d = var_ate_.reshape(len(test_sizes),k).transpose()
    
    return a,b,c,d


def writer(name, file):
    with open('../../results/' + name + '.pkl','wb') as f:
        pkl.dump(file, f)
        
def reader(name):
    with open('../../results/' + name + '.pkl', 'rb') as f:
        df = pkl.load(f)
    return df