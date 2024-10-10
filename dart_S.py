import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
start_date = datetime(1949, 1, 1)
end_date = datetime(1960, 12, 1)
num_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1

# Generate date range
dates = [start_date + timedelta(days=30 * i) for i in range(num_months)]

# Generate fake passenger data
np.random.seed(42)  # For reproducibility
passengers = np.random.randint(100, 600, size=num_months)

# Create DataFrame
data = {
    "Month": [date.strftime("%Y-%m") for date in dates],
    "#Passengers": passengers
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("FakeAirPassengers.csv", index=False)

print(df.head())