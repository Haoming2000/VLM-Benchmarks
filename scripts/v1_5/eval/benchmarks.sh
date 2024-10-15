#!/bin/bash

echo "model path: $1"

## MM-VET
python -m llava.eval.interleave_vqa \
    --model-path $1 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /viscam/projects/SceneAug/haoming/mm-vet \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-interleave-qwen-0.5b-hf.jsonl 

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-interleave-qwen-0.5b-hf.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-interleave-qwen-0.5b-hf.json

cd /sailhome/zhm2023/LLaVA/playground/data/eval/mm-vet
python mm-vet_evaluator.py --mmvet_path ../mm-vet --result_file results/llava-interleave-qwen-0.5b-hf.json --openai_api_key your_openai_key
cd ../../../..


## MME

python -m llava.eval.interleave_vqa \
    --model-path $1 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /viscam/projects/SceneAug/haoming/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-interleave-qwen-0.5b-hf.jsonl \

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-interleave-qwen-0.5b-hf

cd eval_tool
 
python calculation.py --results_dir answers/llava-interleave-qwen-0.5b-hf

cd ../../../../..

## POPE
python -m llava.eval.interleave_vqa \
    --model-path $1 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /viscam/projects/SceneAug/haoming/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-interleave-qwen-0.5b-hf.jsonl \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-interleave-qwen-0.5b-hf.jsonl


# ## VQAv2

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b"
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path liuhaotian/llava-v1.5-7b \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

## Viswiz
python -m llava.eval.interleave_vqa \
    --model-path $1 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /viscam/projects/SceneAug/haoming/vizwiz/test/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-interleave-qwen-0.5b-hf.jsonl \


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-interleave-qwen-0.5b-hf.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-interleave-qwen-0.5b-hf.json



# ## MMBench
# SPLIT="mmbench_dev_20230712"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file /viscam/projects/SceneAug/haoming/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file /viscam/projects/SceneAug/haoming/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#     --experiment llava-v1.5-7b


## AMBER

python -m llava.eval.interleave_vqa \
    --model-path llava-hf/llava-interleave-qwen-0.5b-hf \
    --question-file ./AMBER-master/data/query/query_all.jsonl \
    --image-folder /viscam/projects/SceneAug/haoming/amber \
    --answers-file ./AMBER-master/results/llava-interleave-qwen-0.5b-hf.jsonl \

cd AMBER-master
python inference.py --inference_data results/llava-interleave-qwen-0.5b-hf.jsonl --evaluation_type a
cd ..
