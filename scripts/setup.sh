#!/bin/bash

# scrach dir
sudo apt-get update
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse
cd ~
mkdir data
mkdir checkpoints
gcsfuse --implicit-dirs --file-mode=777 --dir-mode=777 thoughtbubbles_data ./data
gcsfuse --implicit-dirs --file-mode=777 --dir-mode=777 thoughtbubbles_checkpoints_ue1 ./checkpoints

# repo
curl -LsSf https://astral.sh/uv/install.sh | sh
case ":${PATH}:" in
    *:"$HOME/.local/bin":*)
        ;;
    *)
        # Prepending path in case a system-installed binary needs to be overridden
        export PATH="$HOME/.local/bin:$PATH"
        ;;
esac
git clone https://github.com/Jemoka/fork-xla.git
pushd ./fork-xla
uv sync
popd

# transparent hugepages
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# magic?!

#set -euo pipefail

#pushd ~/fork-xla

#uv run main.py   test \
 #-vv \
 #--distributed \
 #--warm_start "$HOME/checkpoints/test/recovery" \
#--data_file "$HOME/data/recipes/pretrain.toml" \
#--plan regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular regular \
 #--flops_promised 918e12 \
#--block_size 512 \
#--n_head 16 \
#--out_dir "$HOME/checkpoints" \
#--validation_interval 2048 \
#--checkpoint_interval 10240 \
#--per_device_batch_size 32 \
#--validation_steps 2048 \
#--shard_into 4 \
#--report_interval 1

#popd
