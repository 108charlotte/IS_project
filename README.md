# Predicting Prisoner's Dilemma Behavior with Machine Learning
## Instructions for running locally: 
Run the following commands in your terminal to set up a virtual environment using the provided requirements.txt. 
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

And you're ready to go! 

## Repository map: 
*Personality model*: 
- data_pdf_faces_2017-03-19_16-36(2).csv: data before processing (file origin: https://osf.io/d5czr/files/7hv6t, from paper: https://www.nature.com/articles/s41598-019-38796-0)
- data_with_computer_decisions.csv: processed data (includes labels for computer decisions, not just scores)
- train_personality_nn.py: code for data processing and hyperparameter tuning specific to the personality neural network
- training_functions.py: contains standard code for training a neural network, given certain inputs
- models.py: contains the architectural specifications for a basic neural network built with pytorch

*Emotion model*: 
- Fields.csv: emotion data for prisoner's dilemma decisions from https://zenodo.org/records/4321821
- train_emotion_nn.py: code for data processing and training specific to the emotion neural network
- training_functions.py: contains standard code for training a neural network, given certain inputs
- models.py: contains the architectural specifications for a basic neural network built with pytorch

*Predictive Coding RNN (strategy) model*: 
- data_pdf_faces_2017-03-19_16-36(2).csv: data before processing (file origin: https://osf.io/d5czr/files/7hv6t, from paper: https://www.nature.com/articles/s41598-019-38796-0)
- data_with_computer_decisions.csv: processed data (includes labels for computer decisions, not just scores)
- train_pcrnn.py: code for data processing and training specific to the PCRNN
- pcrnn.py: classes defining the functionality and architecture of a predictive coding RNN, using the predictive coding neural network from https://arxiv.org/html/2506.06332v1

*Project evolution*: 
- RNN: I originally developed a simple RNN to model strategy, to test if the data had patterns a model could learn at a base level
    - train_rnn.py: trains a basic neural network on the same data the PCRNN later trains on
    - models.py: contains the code for a basic pytorch recurrent neural network