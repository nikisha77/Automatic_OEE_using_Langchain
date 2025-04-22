import pandas as pd
import numpy as np
from datetime import datetime, timedelta

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
device_ids = [f"D{i:02d}" for i in range(1, 11)]  # 10 devices
locations = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai"]  # 5 locations
shifts_per_day = 2  # Assuming two shifts per day
scheduled_time_per_shift_hrs = 10

ideal_cycle_times = {device: np.random.uniform(1, 3) for device in device_ids}

data = []

current_date = start_date
while current_date <= end_date:
    for shift in range(shifts_per_day):
        for device_id in device_ids:
            location = np.random.choice(locations)

            scheduled_hrs = scheduled_time_per_shift_hrs
            downtime_hrs = np.random.uniform(0, scheduled_hrs * 0.15)
            running_time_hrs = scheduled_hrs - downtime_hrs

            ideal_cycle_sec = ideal_cycle_times[device_id]

            ideal_units_possible = (running_time_hrs * 3600) / ideal_cycle_sec
            total_units_produced = max(
                0, int(ideal_units_possible * np.random.uniform(0.8, 1.0))
            )  # Performance factor ~80-100%

            # Good units calculation (based on total units, with some variability in quality)
            good_units_produced = max(
                0, int(total_units_produced * np.random.uniform(0.95, 1.0))
            )  # Quality factor ~95-100%

            data.append(
                {
                    "Timestamp": current_date + timedelta(hours=shift * 12),
                    "Date": current_date,
                    "Shift": shift + 1,
                    "Device_ID": device_id,
                    "Location": location,
                    "Scheduled_Run_Time_Hrs": scheduled_hrs,
                    "Downtime_Hrs": round(downtime_hrs, 2),
                    "Total_Units_Produced": total_units_produced,
                    "Good_Units_Produced": good_units_produced,
                    "Ideal_Cycle_Time_Sec": round(ideal_cycle_sec, 2),
                }
            )

    current_date += timedelta(days=1)

df = pd.DataFrame(data)

# Save to XLSX file
output_filename = "oee_data.xlsx"
df.to_excel(output_filename, index=False)

print(f"Synthetic OEE data generated and saved to '{output_filename}'")
print(f"Data shape: {df.shape}")
print("\nSample Data:")
print(df.head())
