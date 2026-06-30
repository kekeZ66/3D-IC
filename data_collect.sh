#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export DATA_PYTHON='/data2/kmz/envs/ovmm-data/bin/python'
session_name='cu2'

# 新建会话与窗口0（初始只有一个 pane，记下它的 pane_id）
tmux new-session -d -s "${session_name}" -n 0
p0=$(tmux list-panes -t "${session_name}:0" -F '#{pane_id}' | head -n1)

# 在 p0 下方竖切，得到底部 pane 的 id
p1=$(tmux split-window -v -t "${p0}" -P -F '#{pane_id}')

# 在 p0 右侧横切，得到右上 pane 的 id
p2=$(tmux split-window -h -t "${p0}" -P -F '#{pane_id}')

# 在 p1（左下）右侧横切，得到右下 pane 的 id
p3=$(tmux split-window -h -t "${p1}" -P -F '#{pane_id}')

# 排版
tmux select-layout -t "${session_name}:0" tiled
tmux setw -t "${session_name}" remain-on-exit on

# 分别在四个 pane 中执行命令（记得 C-m 回车）
tmux send-keys -t "${p0}" \
  "CUDA_VISIBLE_DEVICES=2 ${DATA_PYTHON} projects/habitat_ovmm/eval_baselines_agent.py --scene_id 103997424_171030444" C-m
tmux send-keys -t "${p2}" \
  "CUDA_VISIBLE_DEVICES=2 ${DATA_PYTHON} projects/habitat_ovmm/eval_baselines_agent.py --scene_id 103997460_171030507" C-m
# tmux send-keys -t "${p1}" \
#   "CUDA_VISIBLE_DEVICES=3 ${DATA_PYTHON} projects/habitat_ovmm/eval_baselines_agent.py --scene_id 102816216" C-m
# tmux send-keys -t "${p3}" \
#   "CUDA_VISIBLE_DEVICES=3 ${DATA_PYTHON} projects/habitat_ovmm/eval_baselines_agent.py --scene_id 102817200" C-m

# 进入会话
tmux attach -t "${session_name}"





