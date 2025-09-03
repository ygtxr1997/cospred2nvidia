MODEL_TIME="2025-08-24_22-51-05"
ITERATION="000010000"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_cchi_v7_replay.zarr"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_orange_random_v2.zarr"
INPUT_DATASET_PATH="./datasets/pusht/pusht_256_val.zarr"
INPUT_DATASET_INDEX=300
INPUT_VIDEO_FRAME_INDEX=0


python examples/video2world_pusht.py \
  --model_size 2B \
  --dit_path "checkpoints/cosmos_predict2/debug/predict2_video2world_2b_action_conditioned_training_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
  --input_video ${INPUT_DATASET_PATH} \
  --dataset_index ${INPUT_DATASET_INDEX} \
  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
  --num_conditional_frames 1 \
  --save_path output/pusht_base_${ITERATION}_${INPUT_DATASET_INDEX}.mp4 \
  --guidance 0 \
  --seed 0 \
  --autoregressive  \
  --disable_guardrail \
  --disable_prompt_refiner
