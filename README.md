### MInf Project - Year 1

To rerun the codebase, please follows the below instructions:

```
pip install -r requirements.txt # install the requirement package

cd hh_dataset

bash download.sh # Downloading the targeted 30 datasets and unzip it

cd ..

cd ftw_model

# Run the Ens-CNN-LSTM experiments with default dataset
bash run.sh

# Run the reconstructed Joint-CNN-LSTM experiments with default dataset
bash joint.sh 

# Run the Ens-LSTM experiments with default dataset
bash run_lstm.sh 

cd ..

# Run the Deep CASAS model
python data_hh.py # Build the feature for Deep CASAS model
python train.py # Start training

## Run the AL-smarthome experiment
cd code
cd AL-smarthome
bash run.sh
```
