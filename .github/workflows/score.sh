# !/bin/bash
set -e

pwd
echo "Current working directory: $(pwd)"

# setup 
./setup.sh
# lint code
./lint.sh
# reproduce results
./reproduce.sh