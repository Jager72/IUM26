import kagglehub
from kagglehub import KaggleDatasetAdapter

path = kagglehub.dataset_download("ihormuliar/starbucks-customer-data", output_dir=".\\Data\\")

print(path)

