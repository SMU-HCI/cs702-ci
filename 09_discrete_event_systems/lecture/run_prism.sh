#!/bin/bash

# To run the following commands, make sure you have the PRISM model checker installed and available in your PATH.
# For this, you can run `bash install_prism.sh` from the root of the repository.

# DTMC model: engagement.pm
# Checks if the PRISM model is syntactically correct.
prism engagement.pm

# Forward simulate the model for 10 steps and save the trace to a file.
prism engagement.pm -simpath 10 engagement.trace

# CTMC model: engagement_timed.pm
# Check if the CTMC model is syntactically correct.
prism engagement_timed.pm

# Forward simulate the CTMC model for 60 time units and save the trace to a file.
prism engagement_timed.pm -simpath time=60 engagement_timed.trace