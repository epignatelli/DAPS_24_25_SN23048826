# !/bin/bash
set -e

pwd
echo "Current working directory: $(pwd)"

# setup 
.github/workflows/setup.sh
# lint code
.github/workflows/lint.sh
# reproduce results
.github/workflows/reproduce.sh