import argparse
import numpy as np
import rasterio
import numpy as np
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import pandas as pd
import pickle

from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from PIL import Image
from torch import nn


"""
Uses raw or best finetuned DinoV2 model to predict severe deprivation from a satellite image
As before the DinoV2 model has an addtional linear layer with sigmoid activation which outputs a 99 dimension vector
The model finally feeds this 99 dimension vector through a standard scaler and ridge regression model -
in order to output severe deprivation
The model returns the mae (for spatial data we average the mae over the folds)

Satellite imagery is saved in the following file structure
Sub directories should be of the form country code + year + satellite
Filenames are the CENTROID_ID

- imagery parent directory
--- ET2018S2
------ ET2000000090.tif
------ ET2000000213.tif
------ ...
--- RW2018S2
------ ...
--- ...
"""


def evaluate(fold, use_checkpoint = False, imagery_path = None, imagery_source = None, mode = 'temporal'):
    """
    Evaluates the DinoV2 on the temporal/spatial train and test data
    Returns the mae
    
    Parameters:
        fold (integer): fold number
        use_checkpoint (boolean): whether to use best finetuned model
        imagery_path (string): parent directory of imagery
        imagery_source (string): Landsat (L) or Sentinel (S)
        mode (string): temporal or spatial
        
    Returns:
        mae (float): Mean absolute error
    """

    # load raw or best finetuned model
    model_par_dir = r'modelling/dino/model/'
    if use_checkpoint:
        if mode == 'temporal':
            checkpoint = f'{model_par_dir}dinov2_vitb14_temporal_best_{imagery_source}.pth'
        elif mode == 'spatial':
            checkpoint = f'{model_par_dir}dinov2_vitb14_{fold}_all_cluster_best_{imagery_source}.pth'
        else:
            raise Exception()
    model_output_dim = 768

    if imagery_source == 'L':
        normalization = 30000.
        transform_dim = 336
    elif imagery_source == 'S':
        normalization = 3000.
        transform_dim = 994

    # load train and test data from specified fold, or from pre/post 2020 data
    data_folder = r'survey_processing/processed_data/'
    if mode == 'spatial':
        train_df = pd.read_csv(f'{data_folder}train_fold_{fold}.csv')
        test_df = pd.read_csv(f'{data_folder}test_fold_{fold}.csv')
    elif mode == 'temporal':
        train_df = pd.read_csv(f'{data_folder}before_2020.csv')
        test_df = pd.read_csv(f'{data_folder}after_2020.csv')

    # store file paths of all available imagery in following list
    available_imagery = []
    for d in os.listdir(imagery_path):
        # d[-2] will either be S or L, refer to top comment for more information
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))
    
    # gets filename of each image without the .fileformat
    available_centroids = [f.split('/')[-1][:-4] for f in available_imagery]
    # filter df to remove rows with no corresponding satellite image
    train_df = train_df[train_df['CENTROID_ID'].isin(available_centroids)]
    test_df = test_df[test_df['CENTROID_ID'].isin(available_centroids)]


    def filter_contains(query):
        """
        Returns a list of items that contain the given query substring.
        
        Parameters:
            query (str): The substring to search for in each item of the list.
            
        Returns:
            list of str: A list containing all items that have the query substring.
        """

        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item
    
    # add file path of satellite imagery corresponding to each row
    # remove rows of data with no value for orphaned
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    train_df = train_df[train_df['orphaned'].notna()]
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)
    test_df = test_df[test_df['orphaned'].notna()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')


    class ViTForRegression(nn.Module):
        """
        Parent class is nn.Module (i.e DinoV2 model)
        Adds additional linear layer with sigmoid activation function in order to get output of length len(predict_target)
        """
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(model_output_dim, 2)  # Output one continuous variable

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # We use the last hidden state
            return torch.sigmoid(self.regression_head(outputs))
        
        def forward_encoder(self, pixel_values):
            return self.base_model(pixel_values)
        

    # load best finetuned model if necessary
    model = ViTForRegression(base_model)
    if use_checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    
    
    def get_features(df):
        """
        Given a DataFrame, feed the satellite images through the DinoV2 model
        Save the feature vectors in a list

        Parameters:
            df (pd.DataFrame): dataframe containing 
        
        Returns:
            dino_features (list): list of feature vectors output from DinoV2 modelS
        """
        dino_features = []
        
        transform = transforms.Compose([
            transforms.Resize((transform_dim, transform_dim)),  # Resize the image to the input size expected by the model
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
        ])
        
        for idx in tqdm(range(len(df))):
            # convert image to tensor
            image = load_and_preprocess_image(df.iloc[idx]['imagery_path'], normalization)
            image_tensor = transform(Image.fromarray(image))

            # evaluate model, with no gradient calculations to increase speed
            with torch.no_grad():
                features = model.forward_encoder(torch.stack([image_tensor]).to(device))
            for f in features:
                dino_features.append(f.cpu())

                # with open(filename, 'wb') as file:
                #     pickle.dump(dino_features, file)
        return dino_features
    
    # get feature vectors corresponding to each image in train and test df
    train_sat_features = get_features(train_df)
    test_sat_features = get_features(test_df)

    # convert list of feature vectors to DataFrame
    train_features_df = pd.DataFrame([f.tolist() for f in train_sat_features])
    test_features_df = pd.DataFrame([f.tolist() for f in test_sat_features])

    train_target = train_df['orphaned']
    test_target = test_df['orphaned']
    # Create a list of alphas to consider for RidgeCV
    alphas = np.logspace(-6, 6, 13)

    # Creating the pipeline with StandardScaler and RidgeCV
    # RidgeCV is initialized with a list of alphas
    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    # Fit the model
    pipeline.fit(train_features_df, train_target)

    # Save model params in a json file
    weights = pipeline[1].coef_
    intercept = pipeline[1].intercept_
    params_dict = {"weights" : weights, "intercept" : intercept}
    file_path = f"modelling/dino/model/ridge_regr_weights_fold_{fold}"
    with open(file_path, "wb") as f:
        pickle.dump(params_dict, f)

    # Predict on test data
    y_pred = pipeline.predict(test_features_df)

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_target, y_pred)

    print("Mean Absolute Error on Test Set:", mae)
    return mae


def load_and_preprocess_image(path, normalization):
    with rasterio.open(path) as src:
        # Read the specific bands (4, 3, 2 for RGB)
        r = src.read(4)  # Band 4 for Red
        g = src.read(3)  # Band 3 for Green
        b = src.read(2)  # Band 2 for Blue
        # Stack and normalize the bands
        img = np.dstack((r, g, b))
        img = img / normalization*255.  # Normalize to [0, 1] (if required)
        
    img = np.nan_to_num(img, nan=0, posinf=255, neginf=0)
    img = np.clip(img, 0, 255)  # Clip values to be within the 0-255 range
    
    return img.astype(np.uint8)  # Convert to uint8


# handle command line inputs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--mode', type=str, default='temporal', help='Evaluating temporal model or spatial model')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint file. If not, use raw model.')

    args = parser.parse_args()
    maes = []
    if args.mode == 'temporal':
        print(evaluate(1, args.use_checkpoint, args.imagery_path, args.imagery_source, args.mode))
    
    else:
        # for the spatial data, for each fold, we choose the best finetuned model (or just the raw model)
        # we evaluate the model and average the maes over the folds
        for i in range(5):
            fold = i + 1
            mae = evaluate(fold, args.use_checkpoint,args.imagery_path, args.imagery_source, args.mode)
            maes.append(mae)
        print(np.mean(maes), np.std(maes)/np.sqrt(5))
