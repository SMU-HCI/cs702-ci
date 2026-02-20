#!/bin/bash

# To run the following commands, make sure you have the PRISM model checker installed and available in your PATH.
# For this, you can run `bash install_prism.sh` from the root of the repository.

# To verify a specific property, run prism from command line (e.g., `prism engagement.pm -pf 'P=? [ F "converted" ]'`)

# DTMC model: engagement.pm
prism engagement.pm engagement_reachability.pctl
prism engagement.pm engagement_invariance.pctl
prism engagement.pm engagement_verification.pctl

# MDP model: activity_agent.pm
prism activity_agent.pm activity_agent_reachability.pctl
prism activity_agent.pm activity_agent_invariance.pctl
prism activity_agent.pm activity_agent_reward.pctl