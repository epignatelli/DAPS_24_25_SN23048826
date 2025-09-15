# lint code
source activate daps-final
pip install pylint
MESSAGE=$(pylint -ry $(git ls-files '*.py') ||:)
