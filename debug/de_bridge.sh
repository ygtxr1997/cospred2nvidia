MODEL_TIME="2025-08-13_20-39-02"
ITERATION="000090000"
INPUT_VIDEO_INDEX=3
INPUT_VIDEO_FRAME_INDEX=0

#python examples/video2world_action_lora.py \
#  --model_size 2B \
#  --dit_path "checkpoints/cosmos_predict2/debug/predict2_video2world_2b_action_conditioned_training_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
#  --input_video datasets/bridge/videos/test/${INPUT_VIDEO_INDEX}/rgb.mp4 \
#  --input_annotation datasets/bridge/annotation/test/${INPUT_VIDEO_INDEX}.json \
#  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
#  --num_conditional_frames 1 \
#  --save_path output/${INPUT_VIDEO_INDEX}_lora_${ITERATION}_${INPUT_VIDEO_FRAME_INDEX}.mp4 \
#  --guidance 0 \
#  --seed 0 \
#  --disable_guardrail \
#  --disable_prompt_refiner \
#  --use_lora

python examples/video2world_action.py \
  --model_size 2B \
  --dit_path "checkpoints/cosmos_predict2/debug/predict2_video2world_2b_action_conditioned_training_${MODEL_TIME}/checkpoints/model/iter_${ITERATION}.pt" \
  --input_video datasets/bridge/videos/test/${INPUT_VIDEO_INDEX}/rgb.mp4 \
  --input_annotation datasets/bridge/annotation/test/${INPUT_VIDEO_INDEX}.json \
  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
  --num_conditional_frames 1 \
  --save_path output/${INPUT_VIDEO_INDEX}_base_${ITERATION}_${INPUT_VIDEO_FRAME_INDEX}.mp4 \
  --guidance 0 \
  --seed 0 \
  --disable_guardrail \
  --disable_prompt_refiner


#python examples/video2world_action.py \
#  --model_size 2B \
#  --dit_path "checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt" \
#  --input_video datasets/bridge/videos/test/${INPUT_VIDEO_INDEX}/rgb.mp4 \
#  --input_annotation datasets/bridge/annotation/test/${INPUT_VIDEO_INDEX}.json \
#  --frame_index ${INPUT_VIDEO_FRAME_INDEX} \
#  --num_conditional_frames 1 \
#  --save_path output/${INPUT_VIDEO_INDEX}_example_${INPUT_VIDEO_FRAME_INDEX}.mp4  \
#  --guidance 0 \
#  --seed 0 \
#  --disable_guardrail \
#  --disable_prompt_refiner


#cp "datasets/bridge/videos/test/${INPUT_VIDEO_INDEX}/rgb.mp4" "output/${INPUT_VIDEO_INDEX}_gt.mp4"
