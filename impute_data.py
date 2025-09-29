import numpy as np
import pandas as pd
from GAIN_model import GAIN
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

df = pd.read_csv('./Data/data_merge_part.csv') 
data_with_nan = df.values.astype(np.float32)
model_path = "./model"
mask = 1 - np.isnan(data_with_nan)

x_incomplete = np.nan_to_num(data_with_nan, nan=0.0)
n, d = x_incomplete.shape

scaler = MinMaxScaler()
scaler.fit(x_incomplete) 
x_scaled = scaler.transform(x_incomplete)


gain = GAIN(data_dim=d, hint_rate=0.9, alpha=10.0)
gain.train(x_scaled, mask, batch_size=128, epochs=1000, verbose=1)

x_imputed_scaled = gain.impute(x_scaled, mask)

x_imputed = scaler.inverse_transform(x_imputed_scaled)

print("Dữ liệu sau khi được điền:")

imputed_df = pd.DataFrame(x_imputed, columns=df.columns)
print(imputed_df.head(10))
output_filename = './Data/imputed_data_merge_part2.csv'

imputed_df.to_csv(output_filename, index=False)
print(f"\nLưu dữ liệu đã điền vào file: '{output_filename}'")
gain.save_model(model_path)

