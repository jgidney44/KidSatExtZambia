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
Uses raw or best finetuned DinoV2 model to predict orphanhood from a satellite imagery
Input: folder of satellite imagery, dataframe of coords of satellite imagery
Output: orphanhood predictions adjoined to the dataframe of coords

Only for spatial predictions, shouldn't be too hard to add code for temporal predictions

dataframe columns are: name, lat, lon
image file names are: name.tif
"""


def predict(use_checkpoint, imagery_path, data_path, imagery_source, fold):
    # load raw or best finetuned model
    model_par_dir = r'modelling/dino/model/'
     
    if use_checkpoint:
        checkpoint = f'{model_par_dir}dinov2_vitb14_{fold}_all_cluster_best_{imagery_source}.pth'
    model_output_dim = 768

    if imagery_source == 'L':
        normalization = 30000.
        transform_dim = 336
    elif imagery_source == 'S':
        normalization = 3000.
        transform_dim = 994

    # load coords df and imagery
    df = pd.read_csv(data_path)

    # make sure identifier is a string
    df["name"] = df["name"].astype(str)

    # add column indicating whether the data is in/out of sample
    train_df_for_fold = pd.read_csv(f"survey_processing/processed_data/train_fold_{fold}.csv")
    train_data_ids = train_df_for_fold["CENTROID_ID"].to_list()
    df["in_sample"] = False
    df.loc[df["name"].isin(train_data_ids), "in_sample"] = True

    # remove rows from df if we don't have the images
    available_imagery = []
    for country_dir in os.listdir(imagery_path):
        available_imagery.extend([f.split('/')[-1][:-4] for f in os.listdir(f"{imagery_path}/{country_dir}")])

    df = df[df["name"].isin(available_imagery)]
    print(df.shape)

    # add imagery file path column
    # par_dir is of the form ZM2018
    # need to change this for full generality for landsat imagery
    par_dir = df["name"].str.slice(0, 6) + "S2"
    df["imagery_path"] = imagery_path + "/" + par_dir + "/" +  df["name"] + ".tif"

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

        return dino_features

    print(df.head())
    print(torch.cuda.is_available())

    # get feature vectors corresponding to each image in train and test df
    sat_features = get_features(df)

    # convert list of feature vectors to DataFrame
    features_df = pd.DataFrame([f.tolist() for f in sat_features])

    # Create a list of alphas to consider for RidgeCV
    alphas = np.logspace(-6, 6, 13)

    # Creating the pipeline with StandardScaler and RidgeCV
    # RidgeCV is initialized with a list of alphas
    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    # Load the fitted model
    # In order to do this, we have to fit the model on any data of the correct size, THEN change the model weights

    pipeline.fit(features_df[:5], [0, 0, 0, 0, 0])
    # now we can change the coefficients
    with open(f"modelling/dino/model/ridge_regr_weights_fold_{fold}", "rb") as f:
        params_dict = pickle.load(f)
    pipeline[1].coef_ = params_dict["weights"]
    pipeline[1].intercept_ = params_dict["intercept"]

    # Predict on test data
    preds = pipeline.predict(features_df)

    # add predictions to coords df, subset df
    df["orphaned"] = preds
    df = df[["name", "lat", "lon", "orphaned", "in_sample"]]

    print(df.head())

    # save to file
    df.to_csv(f"prediction_data/orphanhood_predictions_fold_{fold}.csv", index=False)


def load_and_preprocess_image(path, normalization):
    with rasterio.open(path) as src:
        # Read the specific bands for RGB
        try: # if downloaded rgb_only = FALSE
            r = src.read(4) 
            g = src.read(3)
            b = src.read(2)
        except: # if downloaded rgb_only = TRUE
            b = src.read(1)
            g = src.read(2)
            r = src.read(3)
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
    parser.add_argument('--imagery_path',type=str, help='The parent directory of all imagery')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint file. If not, use raw model.')
    parser.add_argument("--data_path", type=str, help="The parent directory of the imagery coordinates")
    args = parser.parse_args()
    
    for i in range(1, 6):
        predict(args.use_checkpoint, args.imagery_path, args.data_path, args.imagery_source, i)
