MODEL_TIME="2025-10-15_16-48-08"
ITERATION="000018000"
INPUT_DATASET_PATH="/home/geyuan/local_soft/TCL/1009_spoon_pick_place"
INPUT_DATASET_INDICES=(100)  #(0 250 500)  #(300 100 0)
INPUT_VIDEO_FRAME_INDEX=0


for INPUT_DATASET_INDEX in "${INPUT_DATASET_INDICES[@]}"; do

  python examples/video2world_tcl.py \
  --model_size 2B \
  --load_ema  \
  --dit_path "checkpoints/cosmos_predict2/debug/cospred2_2b_expert_tcl_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
  --input_video ${INPUT_DATASET_PATH} \
  --dataset_index ${INPUT_DATASET_INDEX} \
  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
  --num_conditional_frames 1 \
  --chunk_size 20  \
  --save_path output/tcl_base_${ITERATION}_${INPUT_DATASET_INDEX}.mp4 \
  --guidance 0 \
  --seed 0 \
  --autoregressive  \
  --disable_guardrail \
  --disable_prompt_refiner

done

echo "All dataset indices processed."
