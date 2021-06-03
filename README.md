https://us04web.zoom.us/j/3719571814?pwd=QlYzUDNTaFBIWEhYQWd5bEwvbGV0QT09

df = pd.DataFrame([test_sizes, mean_ate_[0], mean_ate_[1],mean_ate_[2], mean_ate_[3], 
                   mean_ate_[4], mean_ate_[5],mean_ate_[6], mean_ate_[7],
                   mse_ate_[0], mse_ate_[1],mse_ate_[2], mse_ate_[3],
                   mse_ate_[4], mse_ate_[5],mse_ate_[6], mse_ate_[7],
                   bias_ate_[0], bias_ate_[1], bias_ate_[2], bias_ate_[3], 
                   bias_ate_[4], bias_ate_[5], bias_ate_[6], bias_ate_[7], 
                   var_ate_[0], var_ate_[1],var_ate_[2], var_ate_[3], var_ate_[4], var_ate_[5],var_ate_[6], var_ate_[7], 
                   MSE_, bias_, var_]).transpose().set_index(0)
df.columns=['Mean1', 'Mean2', 'Mean3', 'Mean4', 'Mean5', 'Mean6', 'Mean7', 'Mean8',
            'MSE1', 'MSE2','MSE3', 'MSE4', 'MSE5', 'MSE6','MSE7', 'MSE8', 
            'BIAS1', 'BIAS2','BIAS3', 'BIAS4', 'BIAS5', 'BIAS6','BIAS7', 'BIAS8', 
            'VAR1', 'VAR2','VAR3', 'VAR4', 'VAR5', 'VAR6','VAR7', 'VAR8', 'MSE_TOTAL', 'MSE_T_BIAS', 'MSE_T_VAR']
