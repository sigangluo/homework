DATASET=hate
DATASET_BASE=/mnt/nvm_data/guest24/sigangluo/tweeteval/datasets
TEMPLATEID=0 # 1 2 3
SEED=144 # 145 146 147 148
MODE=shot
SHOT=40 # 0 1 10 20
VERBALIZER=manual #manual #kpt  #soft auto
FILTER=tfidf_filter #none 
KPTWLR=0.0 # 0.06
MAXTOKENSPLIT=-1 # 1
MODEL_NAME_OR_PATH="vinai/bertweet-base"

CUDA_VISIBLE_DEVICES=0 python few_shot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--dataset $DATASET \
--dataset_base_path $DATASET_BASE \
--template_id $TEMPLATEID \
--seed $SEED \
--mode $MODE \
--shot $SHOT \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR