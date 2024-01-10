DATASET=hate
DATASET_BASE=/mnt/nvm_data/guest24/sigangluo/tweeteval/datasets
TEMPLATEID=0 # 1 2 3
SEED=144 # 145 146 147 148
VERBALIZER=kpt #
CALIBRATION="--calibration" # ""
FILTER=tfidf_filter # none
MODEL_NAME_OR_PATH="vinai/bertweet-base"

CUDA_VISIBLE_DEVICES=1 python zero_shot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--dataset $DATASET \
--dataset_base_path $DATASET_BASE \
--template_id $TEMPLATEID \
--seed $SEED \
--verbalizer $VERBALIZER $CALIBRATION \
--filter $FILTER