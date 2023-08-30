#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
#from azureml.core import Workspace, Dataset
#def get_data():
#    subscription_id = '7c03dd83-6b95-43b1-9f53-23dfd07e8803'
#    resource_group = 'AZP-102-Temp_AI-RG'
#    workspace_name = 'AZP-102_Temp_AI_ML_POC'
#
#    workspace = Workspace(subscription_id, resource_group, workspace_name)
#
#    dataset = Dataset.get_by_name(workspace, name='udacityProject3')
#
#    return dataset

def clean_data(data):

    #cleanedData = data.drop_columns(['Make', 'Model', 'Vehicle Class', 'Fuel Type', 'Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)', 'Transmission'])
    df = data.drop(['Make', 'Model', 'Vehicle Class', 'Fuel Type', 'Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)', 'Transmission'], axis=1)
    #df = cleanedData.to_pandas_dataframe()
    # cleaning data
    x_df = df
    y_df = x_df.pop("CO2 Emissions(g/km)")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=1.0, help="Maximum depth of decission tree")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="Minimum of samples per leaf")
    #parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--data', help="Data which should be used for training")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Maximum depth of decission tree:", np.int(args.max_depth))
    run.log("Minimum of samples per leaf:", np.int(args.min_samples_leaf))
    
    print('Max depth:{}, min samples per leaf:{}, used data:{}'.format(args.max_depth, args.min_samples_leaf, args.data))

    #url = 'https://azp102tempaiml6246065252.blob.core.windows.net/azureml-blobstore-63601acb-5e43-43dc-8525-86ce80dd0c51/UI/2023-08-09_061836_UTC/CO2 Emissions.csv'
    #data = TabularDatasetFactory.from_delimited_files(path=url)

    input_dataset = run.input_datasets['input']
    data = pd.read_csv(input_dataset)
    #data = TabularDatasetFactory.from_delimited_files(path=input_dataset)

    x, y = clean_data(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)
    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("r2_score", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()
