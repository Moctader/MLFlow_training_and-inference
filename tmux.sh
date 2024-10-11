#!/bin/bash

SESSION_NAME="arcada_pipelines"
tmux new-session -d -s $SESSION_NAME

# SPLIT THE TERMINAL INTO 5 VERTICAL SECTIONS
tmux split-window -v -t $SESSION_NAME:0
tmux split-window -v -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0.1 
tmux split-window -v -t $SESSION_NAME:0.2 
tmux split-window -v -t $SESSION_NAME:0.3

# START EACH PIPELINE COMPONENT IN A DIFFERENT SECTION
tmux send-keys -t $SESSION_NAME:0.0 'make pipeline.gradual_ingest'
tmux send-keys -t $SESSION_NAME:0.1 'make pipeline.data_refinery' C-m
tmux send-keys -t $SESSION_NAME:0.2 'make pipeline.model_dispatch' C-m
tmux send-keys -t $SESSION_NAME:0.3 'make pipeline.decision_synthesis' C-m
tmux send-keys -t $SESSION_NAME:0 'make pipeline.drift_analysis' C-m

# MAKE SURE EACH SECTION IS OF EQUAL SIZE
tmux select-layout -t $SESSION_NAME even-vertical

tmux select-pane -t $SESSION_NAME:0.0
tmux attach-session -t $SESSION_NAME