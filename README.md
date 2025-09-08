# ELEC0136 DAPS Final Assignment 2024/2025 

This project consists of making an AI pipeline comprising of data acquisition, processing, storage, cleaning, exploratory analysis and model inference so as to predict

## Project Structure
 - main.py , Main File, running it will initiate the entire pipeline and perform all the steps mentioned above
 - Data_Acc/ 
    + data_acc.py , contains all the code for aquiring the datasets via the APIs.
- Data_Exp/ 
    + data_exp.py , contains all the code for exploratory data analysis
- Data_Storage/ 
    + data_storage.py , contains all the code for storing and retreiving data from the MongoDB Atlas Cluster.
- env/
    + environment.yml , Code to create a new conda env called "DapsAssgn" with all necessary modules
    + requirements.txt, All needed modules
- explore/ result plots of exploratory data analysis
- inference/ result plots of attempted LSTM model
- Preproc/ 
    + preproc.py , contains all the code for preprocessing the data before exploratory analysis.
- processing/ result plots of preprocessing stage
    + inference.py , contains all the code for training the LSTM model.
- ML/ 
    + inference.py , contains all the code for training the LSTM model.
 - Utility/ , general purpose utility files
    + utils.py , Contains code for cipher and some general purpose code
    + consts.py , stores the string identifier needed to connect to MongoDB Atlas along with the database and collection names (constants)
    + mongodb.json , stores the username and password in an encrypted form which if decrypted and combined with the string identifier gives us the full connection string.
- README.md , This file

## All Packages needed to run the code
- numpy
- scipy
- pandas
- scikit-learn
- requests
- pymongo
- dnspython
- matplotlib
- seaborn
- datetime
- yfinance
- torch

## Instructions
git clone the repo, go into the project root folder and open terminal.
Run the following code to create a new conda environment named "DapsAssgn" with all the necessary modules.
```bash
conda env create -f env/environment.yml
```
Go back to the project root folder and run the following code to inititiate the entire pipeline and perform all the steps mentioned above.
```bash
python3 main.py
```
