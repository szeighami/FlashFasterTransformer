# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
import argparse
import os
import timeit
import torch
import numpy as np
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.gpt.utils.gpt import GPT, GPTWeights
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
import torch.utils.benchmark as benchmark

def set_default():
    os.environ["context_attn_flash"] = "1"    
    os.environ["context_attn_faster"] = "0"    
    os.environ["process_new_launch"] = "0"    
    os.environ["with_decoder_attn"] = "1"    
    os.environ["with_decoder_ffn"] = "1"    
    os.environ["with_context_ffn"] = "1"    
    os.environ["with_context_attn"] = "1"    

def set_setting(with_flash, with_new_kernel, setting):
    set_default()
    if with_flash:
        os.environ["context_attn_flash"] = "1"    
        os.environ["context_attn_faster"] = "0"    
    else:
        os.environ["context_attn_flash"] = "0"    
        os.environ["context_attn_faster"] = "1"    

    if with_new_kernel:
        os.environ["process_new_launch"] = "1"    
    else:
        os.environ["process_new_launch"] = "0"    

    for value in setting:
        os.environ[value] = "0"    


def run_model(args):
    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    max_batch_size = args.max_batch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    enc = encoder.get_encoder(args.vocab_file, args.merges_file)

    contexts = ['Who are you?'] 
    start_ids = [torch.IntTensor(enc.encode(c)) for c in contexts]

    start_lengths = [len(ids) for ids in start_ids]
    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)

    random_seed = 0
    #random_seed = random.randint(0, 100000)

    # Prepare model.
    gpt = GPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
              max_seq_len, tensor_para_size, pipeline_para_size, lib_path=args.lib_path)
    if not gpt.load(ckpt_path=args.ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    if args.data_type == 'fp16':
        gpt.half()
    elif args.data_type == 'bf16':
        gpt.bfloat16()

    if args.sparse:
        gpt.sparse()

    with torch.no_grad():
        # Generate tokens.
        tokens_batch = gpt(start_ids,
                           start_lengths,
                           output_len,
                           beam_width,
                           top_k,
                           top_p,
                           beam_search_diversity_rate,
                           temperature,
                           len_penalty,
                           repetition_penalty,
                           random_seed,
                           return_output_length,
                           return_cum_log_probs)
        tokens_batch = tokens_batch.cpu().numpy()
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            for beam_id in range(beam_width):
                token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                output = enc.decode(token)
                print(output)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocab size')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=1.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='../models/megatron-models/c-model/345m/1-gpu',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_gpt.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')
    parser.add_argument('--max_batch_size', type=int, default=8,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--is_fix_random_seed', type=bool, default=True,
                        help='is fixing the random seed.')
    parser.add_argument('--sparse', action='store_true', dest='sparse',
                        help='Enable sparse matrix multiplication. (Need SM 8.0 or 8.6 and SPARSITY_SUPPORT=ON)')
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')

    args = parser.parse_args()


    print("\n=============== Arguments ===============")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=========================================\n")


    #run_model(args)

    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    repetition_penalty = args.repetition_penalty
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0


    layer_num = 16
    head_num = 32
    d_model = 4096
    max_seq_len = 2048
    size_per_head = d_model//head_num

    output_lens = [1024]
    #prompt_lens = [2, 16, 128, 1024]
    prompt_lens = [1024]
    batch_sizes = [1]#, 2, 4]
    with_flashs = [False]
    with_new_kernels = [True, False]

    random_seed = 0
    reps = 5
    precisions = [16, 32]

    #gpt.bfloat16()

    res = {'with_new_kernel':[], 'with_flash':[], 'batch_size':[], 'prompt_len':[], 'output_len':[], 'precision':[], 'time':[]}
    save_res = False
    save_file = "res_with_opts_d128.csv"

    os.environ["with_decoder_attn"] = "1"    
    os.environ["with_decoder_ffn"] = "1"    
    os.environ["with_context_ffn"] = "1"    
    os.environ["with_context_attn"] = "1"    

    settings = [[], ["with_decoder_attn"], ["with_decoder_ffn"], ["with_context_ffn"], ["with_context_attn"]]

    for precision in precisions:
        gpt = GPT(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                  max_seq_len, tensor_para_size, pipeline_para_size, lib_path=args.lib_path)
        gpt.load(ckpt_path="NO_CHECKPOINT")
        if precision==16:
            gpt.half()

        for with_flash in with_flashs:
            for with_new_kernel in with_new_kernels:
                for batch_size in batch_sizes:
                    for prompt_len in prompt_lens:
                        for output_len in output_lens:
                            start_ids = [torch.IntTensor([10 for _ in range(prompt_len)]) for _ in range(batch_size)]
                            start_lengths = [len(ids) for ids in start_ids]
                            start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
                            start_lengths = torch.IntTensor(start_lengths)

                            for setting in settings:
                                set_setting(with_flash, with_new_kernel, setting) 
                                if setting == []:
                                    setting_name = "all"
                                else:
                                    setting_name = "_".join(setting)

                                with torch.no_grad():
                                    # Generate tokens.
                                    tokens_batch = gpt(start_ids,
                                                       start_lengths,
                                                       output_len,
                                                       beam_width,
                                                       top_k,
                                                       top_p,
                                                       beam_search_diversity_rate,
                                                       temperature,
                                                       len_penalty,
                                                       repetition_penalty,
                                                       random_seed,
                                                       return_output_length,
                                                       return_cum_log_probs)
                                    t = benchmark.Timer(
                                        stmt='''fn(start_ids,
                                                  start_lengths,
                                                  output_len,
                                                  beam_width,
                                                  top_k,
                                                  top_p,
                                                  beam_search_diversity_rate,
                                                  temperature,
                                                  len_penalty,
                                                  repetition_penalty,
                                                  random_seed,
                                                  return_output_length,
                                                  return_cum_log_probs)''',
                                              globals={'fn':gpt, 
                                                  'start_ids':start_ids,
                                                 'start_lengths':start_lengths,
                                                 'output_len':output_len,
                                                 'beam_width':beam_width,
                                                 'top_k':top_k,
                                                 'top_p':top_p,
                                                 'beam_search_diversity_rate':beam_search_diversity_rate,
                                                 'temperature':temperature,
                                                 'len_penalty':len_penalty,
                                                 'repetition_penalty':repetition_penalty,
                                                 'random_seed':random_seed,
                                                 'return_output_length':return_output_length,
                                                 'return_cum_log_probs':return_cum_log_probs})

                                
                                    time = t.timeit(reps).mean

                                    res['with_flash'].append(with_flash)
                                    res['batch_size'].append(batch_size)
                                    res['prompt_len'].append(prompt_len)
                                    res['output_len'].append(output_len)
                                    res['time'].append(time)
                                    res['precision'].append(precision)
                                    res['setting'].append(setting_name)
                                    res['with_new_kernel'].append(with_new_kernel)

                                    if save_res:
                                        res_df = pd.DataFrame(res)
                                        res_df.iloc[-1:].to_csv(save_file, mode='a', header=not os.path.exists(save_file))
                                    
                                    print(f"with_new_kernel:{with_new_kernel}, with_flash:{with_flash}, batch_size:{batch_size},prompt_len:{prompt_len},output_len:{output_len},precision:{precision},time:{time}")

if __name__ == '__main__':
    main()
