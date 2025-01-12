*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Machine Learning Engineer with Microsoft Azure - Capstone Project

This is the capstone project in the Machine Learning Engineer with Microsoft Azure Nanodegree.

In this project, I will select a public external dataset to train a model using Automated ML and Hyperdrive. We will then compare the performance of these two different methods and deploy the most effective model. Ultimately, the endpoint generated will be utilized to obtain insights regarding predictions.

![](/images/01-capstone-diagram.png)

## Dataset
The dataset used in this project is Breast Cancer Wisconsin (Diagnostic) and it donated on 10/31/1995.

### Overview
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image. A few of the images can be found at http://www.cs.wisc.edu/~street/images/

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

- General information:
    - Dataset Characteristics: Multivariate
    - Subject Area: Health and Medicine
    - Associated Tasks: Classification
    - Feature Type: Real
    - Number of Instances: 569
    - Class distribution: 357 benign, 212 malignant
    - Number of Features: 30
    - All feature values are recoded with four significant digits. 
    - Missing attribute values: none

- Detailed:

|Variable|Name|Role|Type|Description|Units|Missing Values|
|--------|----|----|----|-----------|-----|--------------|
|ID|ID|Categorical|||no|
|Diagnosis|Target|Categorical|||no|
|radius1|Feature|Continuous|||no|
|texture1|Feature|Continuous|||no|
|perimeter1|Feature|Continuous|||no|
|area1|Feature|Continuous|||no|
|smoothness1|Feature|Continuous|||no|
|compactness1|Feature|Continuous|||no|
|concavity1|Feature|Continuous|||no|
|concave_points1|Feature|Continuous|||no|
|symmetry1|Feature|Continuous|||no|
|fractal_dimension1|Feature|Continuous|||no|
|radius2|Feature|Continuous|||no|
|texture2|Feature|Continuous|||no|
|perimeter2|Feature|Continuous|||no|
|area2|Feature|Continuous|||no|
|smoothness2|Feature|Continuous|||no|
|compactness2|Feature|Continuous|||no|
|concavity2|Feature|Continuous|||no|
|concave_points2|Feature|Continuous|||no|
|symmetry2|Feature|Continuous|||no|
|fractal_dimension2|Feature|Continuous|||no|
|radius3|Feature|Continuous|||no|
|texture3|Feature|Continuous|||no|
|perimeter3|Feature|Continuous|||no|

- Additional Variable Information:
    1. ID number
    2. Diagnosis (M = malignant, B = benign)
    3. (3-32) Ten real-valued features are computed for each cell nucleus:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

### Task
The main task is classify the the given details of the FNA image into two classes: Malignant and Benign. All the attributes excluding the ID Number are used for training the model. The column diagnosis is the target variable.

### Access
I just download this dataset from UCI and import/upload from local file to my Azure ML workspace.
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

```python
# TODO: Put your automl settings here
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}

# TODO: Put your automl config here
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="diagnosis",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

- Configuration Settings
    - Experiment Timeout Minutes (20 minutes): Limits the total time that all iterations can take, managing computational costs and preventing indefinite runs.
    - Max Concurrent Iterations (5): Defines the maximum number of iterations executed in parallel to balance speed and resource use.
    - Primary Metric (AUC_weighted): Measures the area under the ROC curve, adjusted for class imbalance, suitable for medical diagnostic purposes.

- AutoML Configuration
    - Compute Target (compute_target): Specifies the compute resources where the AutoML job runs, selected based on availability and expected workload.
    - Task (Classification): The machine learning task is set to classify data points into benign or malignant categories.
    - Training Data (dataset): The dataset contains features and labels for training, provided as an Azure Dataset object.
    - Label Column Name ("diagnosis"): Indicates the column with labels (benign or malignant) in the dataset for prediction.
    - Path (project_folder): Directory path for storing project-related files within the Azure environment.
    - Enable Early Stopping (True): Allows stopping the experiment when there is no improvement in the primary metric, saving resources and time.
    - Featurization ('auto'): Automates data preprocessing to ensure optimal data preparation for training.
    - Debug Log ("automl_errors.log"): Records errors during the AutoML run, facilitating troubleshooting and understanding of issues.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The AutoML experiment run generated VotingEnsemble algorithm as the best model with accuracy of 0.98415. <br>

In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. Unlike a statistical ensemble in statistical mechanics, which is usually infinite, a machine learning ensemble consists of only a concrete finite set of alternative models, but typically allows for much more flexible structure to exist among those alternatives.

- The run details of the AutomatedML run are as below:
![](/images/03-automl-run.png)
![](/images/04-automl-overview.png)



*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

- The metrics of the best run model can be seen as below:
![](/images/05-automl-detailed.png)
![](/images/06-automl-run-detailed.png)
![](/images/07-automl-run-detailed_metric.png)

- Deployment here
![](/images/08-automl-deploy.png)



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Hyperparameter sampling in Azure Machine Learning supports three main methods: Random Sampling, Grid Sampling, and Bayesian Sampling.

In this experiment, the RandomForestClassifier from the sklearn library is used. This meta estimator builds multiple decision trees on varied dataset subsets and uses averaging to enhance prediction accuracy and mitigate overfitting. By default, trees are constructed with bootstrap samples (bootstrap=True), but setting bootstrap=False uses the entire dataset for each tree.

The model introduces randomness by choosing the best feature from a random subset at each split, improving diversity and performance. The dataset is divided into a 70% training set and a 30% test set, using a default random state of 2025.

```python
param_sampling = RandomParameterSampling( {
    '--n_estimator': choice(100, 200, 500, 800, 1000),
    '--min_samples_split': choice(2, 5, 10),
    '--max_features': choice('auto', 'sqrt', 'log2')
})
```

- Key hyperparameters for the RandomForestClassifier include:
    - --n_estimator: This parameter specifies the number of trees in the forest for models like the RandomForestClassifier. You've set it to randomly choose among 100, 200, 500, 800, or 1000 trees. The choice of more trees generally improves model accuracy but also increases computational load and time.
    - --min_samples_split: This determines the minimum number of samples required to split an internal node in a tree. The options provided are 2, 5, or 10. A smaller value may lead to a model that captures more information (and potentially noise) about the data, possibly leading to overfitting. Larger values prevent the model from learning overly complex trees but might underfit the data.
    - --max_features: This parameter controls the number of features to consider when looking for the best split at each node. Your choices are 'auto', 'sqrt', or 'log2':
        - 'auto': This will simply use all features which makes the tree building process consider every feature at every split.
        - 'sqrt': This option uses the square root of the total number of features at each split.
        - 'log2': This uses the logarithm base two of the feature count at each split.




### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The Hyperdrive tuned best model generated an accuracy of ```0.9590643274853801``` with the following the configurations:
```
- n_estimators: 100
- min_samples_split: 5
- max_features: log2
- Accuracy 0.9590643274853801
```

All the iteration/child runs of the experiment with different hyperparamters are:
![](/images/09-hyperdriver-run.png)
![](/images/10-hyperdriver-run-output.png)

The hyperdrive tuned best model details can be given as:
![](/images/11-hyperdriver-bestrun.png)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Deploying the optimal model enables interaction with the HTTP API service through POST requests.

The image below displays the real-time endpoint established following the deployment of the optimal model:

![](/images/97-deployment-1.png)


The `test_endpoint.py` script facilitates making a POST request to predict the labels of provided records. This script includes the necessary data payload for the HTTP request:

![](/images/98-deployment-endpoint.png)

Once the endpoint is operational, it generates a scoring URI and a secret key.
These details, the scoring URI and secret key, must be incorporated into the endpoint.py script.

![](/images/99-deployment-endpoint-2.png)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

Youtube link: [https://youtu.be/LWCzYraI1z8](https://youtu.be/LWCzYraI1z8)

## Standout Suggestions
- Prevent Overfitting: To avoid overfitting in models:
    - Increase training data to reduce bias.
    - Prevent target leakage.
    - Reduce feature count.
    - Apply regularization and hyperparameter tuning.
    - Implement model complexity limits.
    - Use cross-validation, especially in automated ML settings where some measures are applied by default.
    - For more details, see: Managing ML Pitfalls - Microsoft Azure
- Expand Hyperparameter Exploration: Utilize a broader range of hyperparameters in the scikit-learn pipeline to optimize model performance.
- Activate Deep Learning: Enable deep learning features during AutoML experiment setup to enhance classification accuracy.
- Data Preprocessing: Conduct feature selection based on their impact on model performance to streamline and enhance model efficacy.