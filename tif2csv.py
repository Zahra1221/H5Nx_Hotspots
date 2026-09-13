import rasterio
from rasterio.transform import xy
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

# Convert WorldClim .tif to .csv
def tif_to_csv(tif_path, output_csv, nodata_to_nan=True):
    with rasterio.open(tif_path) as src:
        data = src.read(1)          # read first band
        transform = src.transform
        nodata = src.nodata

        rows, cols = np.where(data != nodata) if nodata is not None else np.indices(data.shape)

        # Get coordinates of each pixel
        xs, ys = xy(transform, rows, cols)

        values = data[rows, cols]

        if nodata_to_nan and nodata is not None:
            values = np.where(values == nodata, np.nan, values)

        df = pd.DataFrame({
            'longitude': xs,
            'latitude': ys,
            'value': values
        })

        # Optional: drop nodata rows
        df = df.dropna(subset=['value'])

        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}  ({len(df)} rows)")

# Usage
files = os.listdir('./worldclim')
csv_folder = Path('./worldclim_CSV')
csv_folder.mkdir(exist_ok=True)
for item in tqdm(files):
  tif_to_csv('./worldclim/'+item, './worldclim_CSV/'+item[0:-4]+'.csv')

