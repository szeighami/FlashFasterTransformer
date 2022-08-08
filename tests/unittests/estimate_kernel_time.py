import pandas as pd
import os

no_layers = 10
head_num = 32

gpu_name = "3090"
gpu_clock_rate = {"3090":1.35*10**9, "A100":1.08*10**9}
gpu_transfer_rate = {"3090":936.2*2**30, "A100":1550*2**30}

mem_units = {"byte":1, "Kbyte":2**10, "Mbyte":2**20, "Gbyte":2**30}
time_units = {"usecond":1e-6}

#alpha = 0.12#9205722
alpha = 0.10436274819447076
C = 5e-6#14.3437951*1e-6

'''
prompt_lens = [2, 16, 128, 1024]
max_tokenss = [2, 16, 128, 1024]
precisions = [16, 32]
head_sizes = [64, 128]
batch_sizes = [1,2,4,8]
tpks = [1, 2, 4]
tpbs = [64, 128, 256, 512]
'''

prompt_lens = [2, 16, 128, 1024]
#prompt_lens = [1024]
max_tokenss = [1024]
precisions = [16]
head_sizes = [64]
batch_sizes = [1]
tpks = [1, 2, 4]
tpbs = [64, 128, 256, 512]

for precision in precisions:
    for head_size in head_sizes:
        for prompt_len in prompt_lens:
            for max_tokens in max_tokenss:
                for batch_size in batch_sizes:
                    for tpb in tpbs:
                        for tpk in tpks:
                            if precision == 16:
                                tpvs = [2, 4, 8]
                            else:
                                tpvs = [1, 2, 4]
                            for tpv in tpvs:
                                exp_name = f"res_{prompt_len}_{max_tokens}_{batch_size}_{head_size}_{head_num}_{no_layers}_{precision}_{tpk}_{tpv}_{tpb}_{gpu_name}"
                                if not os.path.exists(f"{exp_name}.csv"):
                                    print(f"{exp_name} not found")
                                    continue
                                stats_df = pd.read_csv(f"{exp_name}.csv")

                                no_warp_inst = stats_df[stats_df['Metric Name']=='smsp__inst_executed.max']['Metric Value'].values.astype(int)[0] 
                                warps_per_smsp = stats_df[stats_df['Metric Name']=='smsp__warps_launched.max']['Metric Value'].values.astype(int)[0] 
                                no_inst = no_warp_inst/warps_per_smsp
                               
                                #mem_transfer = stats_df[stats_df['Metric Name']=='sm__sass_data_bytes_mem_global.sum']['Metric Value'].values.astype(float)[0] 
                                #mem_transfer_unit = stats_df[stats_df['Metric Name']=='sm__sass_data_bytes_mem_global.sum']['Metric Unit'].values[0] 
                                #mem_trasfer_bytes = mem_transfer*mem_units[mem_transfer_unit]

                                mem_trasfer_bytes = (precision//8) * batch_size * 4 * head_num*head_size+2*(precision//8)*batch_size*head_num*head_size*(prompt_len)



                                time_underutil = no_inst/(alpha*gpu_clock_rate[gpu_name])
                                time_mem =  mem_trasfer_bytes/gpu_transfer_rate[gpu_name]
                                observed_time = stats_df[stats_df['Metric Name']=='Duration']['Metric Value'].values.astype(float)[0] 
                                observed_time_unit = stats_df[stats_df['Metric Name']=='Duration']['Metric Unit'].values[0] 
                                observed_time_sec = observed_time*time_units[observed_time_unit]

                                alpha_est = no_inst/(gpu_clock_rate[gpu_name]*(observed_time_sec-time_mem-C)) 
                                alpha_est2 = no_inst/(gpu_clock_rate[gpu_name]*(observed_time_sec-C)) 
                                #print(observed_time_sec-time_mem-C, warps_per_smsp, no_inst, alpha_est, alpha_est2)
                                
                                print(exp_name, time_underutil+C, time_mem+C, observed_time_sec)

                                


