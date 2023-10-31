#! /usr/bin/sh

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:18}

# Set the necessary variables
RESOURCE_GROUP="sparkOps-Clfs${suffix}"
RESOURCE_PROVIDER="Microsoft.MachineLearning"
REGION="eastus"
WORKSPACE_NAME="sparkOps-Clfs${suffix}"

# Register the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace and set to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

#create data assets uri_folder
echo "Create a data asset uri_folder:"
az ml data create -f asset.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

 