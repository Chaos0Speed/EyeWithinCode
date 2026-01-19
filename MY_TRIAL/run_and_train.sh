#USE THIS TO CREATE A NEW MODEL AND SIMULTANEOUSLY TRAIN IT

#!/bin/bash

EPOCHS=${1:-50}
MODEL_NAME=${2:-"model"}

clear
cd $HOME/EyeWithinCode/MY_TRIAL
source ../../PythonLibraries/bin/activate
python3 training_datainput.py
python3 preprocess.py
python3 model_creation.py
python3 model_training.py << EOF
u
$EPOCHS
$MODEL_NAME
EOF

echo "Creation And Training Complete"