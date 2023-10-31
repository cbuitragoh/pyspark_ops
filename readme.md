## PySpark Serverless in Azure Machine Learning

<div>
<pre>
<p>This repository showing in a short example
how to setup and run an apache Spark job on serverless
compute in azure ml.

The training_clf.ipynb file can be running on
azure databricks too.

First run setup.sh and then download the
data and convert it to CSV. Source: 
wasbs://nyctlc@azureopendatastorage.blob.core.windows.net/green/puYear=<b>`year`</b>/puMonth=*/*.parquet 
replace `year` with your year of interest. 
(You can collect data from multiple years),
then put it in the data folder.

Then run job.yml and finally monitoring
the execution and results.</p>

</pre>
</div>

