MODEL_TIME="2025-10-04_13-48-52"
ITERATION="000002000"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_256_val.zarr"
INPUT_DATASET_PATH="/home/geyuan/datasets/LIBERO_uva25rss/libero_10"
INPUT_DATASET_INDICES=(0)  #(0 250 500)  #(300 100 0)
INPUT_VIDEO_FRAME_INDEX=0


for INPUT_DATASET_INDEX in "${INPUT_DATASET_INDICES[@]}"; do

  python examples/video2world_libero.py \
  --model_size 2B \
  --load_ema  \
  --dit_path "checkpoints/cosmos_predict2/debug/cospred2_2b_expert_libero_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
  --input_video ${INPUT_DATASET_PATH} \
  --dataset_index ${INPUT_DATASET_INDEX} \
  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
  --num_conditional_frames 1 \
  --chunk_size 20  \
  --save_path output/libero_base_${ITERATION}_${INPUT_DATASET_INDEX}.mp4 \
  --guidance 0 \
  --seed 0 \
  --autoregressive  \
  --disable_guardrail \
  --disable_prompt_refiner

done

echo "All dataset indices processed."
