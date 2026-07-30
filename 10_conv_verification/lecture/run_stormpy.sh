#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# DTMC model: engagement.pm
pixi run python run_stormpy.py engagement.pm engagement_reachability.pctl
pixi run python run_stormpy.py engagement.pm engagement_invariance.pctl
pixi run python run_stormpy.py engagement.pm engagement_verification.pctl

# MDP model: activity_agent.pm
pixi run python run_stormpy.py activity_agent.pm activity_agent_reachability.pctl
pixi run python run_stormpy.py activity_agent.pm activity_agent_invariance.pctl
pixi run python run_stormpy.py activity_agent.pm activity_agent_reward.pctl
