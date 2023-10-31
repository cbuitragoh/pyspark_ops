# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import mlflow
import argparse


def main(args): 
    #setting sparkSession
    spark = create_SparkSession(args.app_name) 

    #load data
    df = get_csvs_df(spark, args.training_data)

    #preprocessing
    cols_to_vector, my_data = preprocess_data(df)

    #final training data
    training_df, test_df = create_vector_assembler(cols_to_vector, my_data)

    #train model
    train_model(training_df)

# setup specs for mlflow experiment
def setup_Experiment(exp_name):
    experiment_name = exp_name
    mlflow.set_experiment(experiment_name)
    mlflow.spark.autolog(disable=True)

# setup sparkSession settings
def create_SparkSession(app_name):
    spark = (
    SparkSession.builder
        .appName(f"{app_name}")
        .config("spark.jars.packages", "org.mlflow.mlflow-spark:1.11.0")
        .master("local[*]")
        .getOrCreate()
    )
    return spark

#loading data
def get_csvs_df(spark, path):
    df = spark.read.csv(path, header=True)

    return df

#drop columns, OneHotEncoder and vectorizer
def preprocess_data(df):
    #Drop unwanted columns
    my_data = df.drop(*["_c0", "lpepPickupDatetime", "lpepDropoffDatetime", "ehailFee"])

    # create object of StringIndexer class and specify input and output column
    for column in my_data.columns:
        SI = StringIndexer(inputCol=column, outputCol=column+'_Index', handleInvalid='skip')
        my_data = SI.fit(my_data).transform(my_data)

    #select imputs cols 
    inputCols = [column for column in my_data.columns if column.endswith('Index')]
    #select output cols
    outputCols = [col+"_OHE" for col in my_data.columns if not col.endswith('Index')]

    # create object and specify input and output column
    OHE = OneHotEncoder(inputCols=inputCols, outputCols=outputCols)

    # transform the data
    my_data = OHE.fit(my_data).transform(my_data)

    #create vector of columns for model
    cols_to_vector =[val[0] for val in my_data.dtypes if val[1] != 'string' and not val[0].startswith("tripType")]

    return cols_to_vector, my_data


def create_vector_assembler(cols_to_vector, data):
    # create a vector assembler object
    assembler = VectorAssembler(inputCols=cols_to_vector,
                            outputCol='features')

    # fill the null values
    my_data = data.fillna(0)

    # transform the data
    final_data = assembler.transform(my_data)

    #Model_Dataframe
    model_df = final_data.select(['features','tripType_Index'])
    model_df = model_df.withColumnRenamed("tripType_Index","label")

    training_df,test_df = model_df.randomSplit([0.75,0.25])

    return training_df,test_df


def train_model(training_df):
    #Create a random forest classifier object
    rf_clf = RandomForestClassifier()
    #Fit the model
    rf_clf_model = rf_clf.fit(training_df)
    log_model_metrics(rf_clf_model)


def log_model_metrics(model):
    #collect and register metrics
    model_summary=model.summary
    mlflow.log_metric("accuracy",model_summary.accuracy)
    mlflow.log_metric("recall",model_summary.weightedRecall)
    mlflow.log_metric("precision",model_summary.weightedPrecision) 
    mlflow.log_metric("f1",model_summary.weightedFMeasure())
    mlflow.log_metric("area under ROC",model_summary.areaUnderROC)
    mlflow.spark.log_model(model, "model")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str, help="Path to training data")
    parser.add_argument("--app_name", dest="app_name",
                        type=str, help="Name of the application")

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")