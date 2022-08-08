import os

no_layers = 10
head_num = 32

gpu_name = "3090"
prompt_lens = [2, 16, 128, 1024]
max_tokenss = [2, 16, 128, 1024]
precisions = [16, 32]
head_sizes = [64, 128]
batch_sizes = [1,2,4,8]
tpks = [1, 2, 4]
tpbs = [64, 128, 256, 512]

'''
prompt_lens = [1024]
max_tokenss = [1024]
precisions = [32]
head_sizes = [64]
batch_sizes = [1]
tpvs = [1]
tpbs = [64]
'''

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
                                print(f"running {exp_name}")
                                os.system(f"ncu --set full --details-all --metrics smsp__thread_inst_executed,smsp__cycles_active,smsp__cycles_elapsed,smsp__average_warp_latency_per_inst_executed,sm__sass_data_bytes_mem_global,smsp__sass_thread_inst_executed,smsp__sass_inst_executed,sm__inst_executed.avg.per_cycle_elapsed,sm__inst_executed,smsp__inst_executed,smsp__thread_inst_executed.avg.per_cycle_elapsed,smsp__thread_inst_executed.avg.per_cycle_active,sm__warps_launched,smsp__warps_launched  -k masked_multihead_attention_kernel_optimized -s 5 -c 1 -f -o {exp_name} ./bin/test_cost_model {prompt_len} {max_tokens} {batch_size} {head_size} {head_num} {no_layers} {precision} {tpk} {tpv} {tpb} &&  ncu -i /workspace/FasterTransformer/build/{exp_name}.ncu-rep --csv > {exp_name}.csv")
