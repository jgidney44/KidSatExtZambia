import download_imagery as di
import pandas as pd
import numpy as np


# let's iterate through the dataframe 3000 at a time
# doing up to 3000
df = pd.read_csv("C:/Users/jgidn/Documents/Summer Project/KidSatExt/imagery_scraping/zambia_grid_center_points.csv")
df = df.iloc[6059:]
df.to_csv("C:/Users/jgidn/Documents/Summer Project/KidSatExt/imagery_scraping/zambia_grid_center_points_snippet.csv", index=False)


file_path_to_csv = "C:/Users/jgidn/Documents/Summer Project/KidSatExt/imagery_scraping/zambia_grid_center_points_snippet.csv"
drive_folder = "Zambia Grid 2018 Landsat"
year = "2018"
sensor = "L8"
range_km = 10 # 10 km
rgb_only = False
di.download_imagery(file_path_to_csv, drive_folder, year, sensor, range_km, rgb_only)