accelerate launch train.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-canny" \
 --output_dir="canny_model" \
 --resolution=512 \
 --train_batch_size=8 \
 --num_train_epochs=100 \
 --checkpointing_steps=840 \
 --validation_steps=840 \
 --enable_xformers_memory_efficient_attention \
 --tracker_project_name="controlnet" \
 --report_to wandb \
 --num_validation_images=1

 
