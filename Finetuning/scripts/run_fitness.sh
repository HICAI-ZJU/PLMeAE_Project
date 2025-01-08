CUDA_VISIBLE_DEVICES=0 fairseq-train ./data \
  --arch fitness_esm1b  \
  --warmup-updates 1 \
  --task fitness \
  --criterion fitness_loss  \
  --max-tokens 2048 \
  --reset-dataloader \
  --required-batch-size-multiple 1 \
  --num-workers 4 \
  --reset-optimizer \
  --optimizer adam \
  --seed 3 \
  --lr 1e-4 \
  --log-interval 1 \
  --validate-after-updates 1 \
  --validate-interval 50 \
  --lr-scheduler inverse_sqrt\
  --user-dir ./Finetuning  \
  --max-epoch  \
  --save-dir  \



