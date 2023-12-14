# Neural networks

Neural network training models built with Tensorflow library:

- sonar ([dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) with accuracy comparison to [data classificator](./../4-data-classification)) and with the confusion matrix output rendered

## How to set up:

Install the packages from the requirements.txt with the following command `pip3 install -r requirements.txt`

## How to run:

For each model, run one of these command in a respective directory:

### sonar

- sonar model: `python3 sonar_model.py`  
  The program will also show the model confusion matrix output.

## Printscreens:

### sonar

Sonar model output:  
![Sonar model output screenshot](./5.1-sonar-model/sonar_model_output.png)

Sonar model output confusion matrix:  
![Sonar model output confusion matrix screenshot](./5.1-sonar-model/sonar_model_confusion_matrix.png)

Sonar output from [Data Classificator](./../4-data-classification)  
![Sonar data classificator output](./5.1-sonar-model/sonar_output_from_data_clasificator.png)

## Authors:

Adam Łuszcz s22994  
Anna Rogala s21487

## Sources:

- https://www.tensorflow.org/?hl=pl
- https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
