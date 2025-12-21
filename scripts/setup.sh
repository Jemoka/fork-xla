#!/bin/bash

# scrach dir
sudo apt-get update
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse
mkdir scratch
gcsfuse --file-mode=777 --dir-mode=777 thoughtbubbles_scratch ./scratch

# repo
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
git clone https://github.com/Jemoka/fork-xla.git
pushd ./fork-xla
uv sync
popd

