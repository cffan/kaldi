#!/usr/bin/env bash

stage=2

. ./cmd.sh
. ./path.sh
. parse_options.sh

export LC_ALL=C

# you might not want to do this for interactive shells.
set -e

# Load libraries on sherlock
ml load imkl
ml gcc/10.1.0

# Number of parallel jobs
nj=8

vocab=/oak/stanford/groups/shenoy/stfan/data/cmudict/dict
G_fst=/oak/stanford/groups/shenoy/stfan/code/nptlrig2/LanguageModelDecoder/examples/speech/s0/openwebtext2/3gram_prune_1e-8/data/lang_test/G.fst

# Prepare dictionary and language model.
# Output L.fst and G.fst
if [ $stage -le 1 ]; then
  local/prepare_dict.sh $vocab data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

   mkdir -p data/lang_nosp_test
   cp -r data/lang_nosp/* data/lang_nosp_test
   #TODO: lm.arpa with correct UNK
   cat /oak/stanford/groups/shenoy/stfan/code/nptlrig2/LanguageModelDecoder/examples/speech/s0/openwebtext2/3gram_prune_1e-8/data/local/lm/lm_pruned.arpa | \
    arpa2fst --disambig-symbol=#0 \
    --read-symbol-table=words.txt - G.fst
   utils/validate_lang.pl --skip-determinization-check data/lang_nosp_test || exit 1;
fi

# Prepare data
# Compute CMVN
steps/compute_cmvn_stats.sh data/train exp/cmvn/train data/train
steps/compute_cmvn_stats.sh data/test exp/cmvn/test data/test
# Compute PCA transform
steps/online/nnet2/train_pca_transform.sh --dim 39 --max-utts 2000 --subsample 2 data/train exp/pca/train

# Train monophone
steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd run.pl --transform-dir exp/pca/train/ data/train/ data/lang_nosp/ exp/mono_pca
# Debug alignment
# show-alignments data/lang_nosp/phones.txt exp/mono_pca/final.mdl ark:"gunzip -c exp/mono_pca/ali.1.gz |" | less
utils/mkgraph.sh data/lang_nosp_test/ exp/mono_pca exp/mono_pca/graph
steps/decode.sh --nj $nj --transform-dir exp/pca/train/ exp/mono_pca/graph/ data/test/ exp/mono_pca/decode_test
grep WER exp/mono_pca/decode_test/wer* | utils/best_wer.sh

# Align
steps/align_si.sh --boost-silence 1.25 --nj $nj --transform-dir exp/pca/train/  data/train data/lang_nosp exp/mono_pca/ exp/mono_pca_ali

# Train triphone
steps/train_deltas.sh --boost-silence 1.25 --transform-dir exp/pca/train/ 2000 10000 data/train data/lang_nosp exp/mono_pca_ali exp/tri1_pca
utils/mkgraph.sh data/lang_nosp_test/ exp/tri1_pca/ exp/tri1_pca/graph
steps/decode.sh --nj $nj --pca-transform-dir exp/pca/train/ exp/tri1_pca/graph/ data/test/ exp/tri1_pca/decode_test
grep WER exp/tri1_pca/decode_test/wer* | utils/best_wer.sh


# Why train on 2k and align on 5k?

# /oak/stanford/groups/shenoy/stfan/code/kaldi/egs/brain2text/s5]$ steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd run.pl ./kaldi_data/t12.2022.08.25/train/ data/lang_nosp exp/mono exp/mono_ali

# steps/train_deltas.sh --boost-silence 1.25 --cmd run.pl 2000 10000 kaldi_data/t12.2022.08.25/train/ data/lang_nosp exp/mono_ali exp/tri1