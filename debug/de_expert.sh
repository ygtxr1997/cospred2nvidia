MODEL_TIME="2025-09-09_22-33-33"
ITERATION="000019000"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_cchi_v7_replay.zarr"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_orange_random_v2.zarr"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_256.zarr"
INPUT_DATASET_PATH="./datasets/pusht/pusht_256_val.zarr"
INPUT_DATASET_INDICES=(0)  #(300 100 0)
INPUT_VIDEO_FRAME_INDEX=0


for INPUT_DATASET_INDEX in "${INPUT_DATASET_INDICES[@]}"; do

  python examples/video2world_expert.py \
    --model_size 2B \
    --dit_path "checkpoints/cosmos_predict2/debug/predict2_video2world_2b_expert_training_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
    --input_video ${INPUT_DATASET_PATH} \
    --dataset_index ${INPUT_DATASET_INDEX} \
    --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
    --save_path output/pusht_expert_dp_${ITERATION}_${INPUT_DATASET_INDEX}.mp4 \
    --guidance 0 \
    --seed 0 \
    --autoregressive  \
    --disable_guardrail \
    --disable_prompt_refiner

done

echo "All dataset indices processed."
