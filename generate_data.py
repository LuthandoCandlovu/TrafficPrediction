import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_traffic_data(days=30, sample_interval=5):
    """
    Generate synthetic traffic data for a single road segment.
    Returns a DataFrame with columns: timestamp, speed, volume, weather_temp, weather_precip
    """
    np.random.seed(42)
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    intervals = int(days * 24 * 60 / sample_interval)
    timestamps = [start_time + timedelta(minutes=i*sample_interval) for i in range(intervals)]

    # Hour of day and day of week features
    hour = np.array([t.hour + t.minute/60 for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])

    # Base speed pattern: sinusoidal with morning and evening peaks
    base_speed = 50 + 10 * np.sin(2 * np.pi * (hour - 7) / 12)   # peaks around 7-9 and 17-19
    # Add day of week effect (lower on weekends)
    weekday_effect = np.where(day_of_week < 5, 0, -15)   # weekend slower
    base_speed += weekday_effect

    # Add random noise and occasional congestion
    noise = np.random.normal(0, 5, intervals)
    congestion_events = np.random.rand(intervals) < 0.05   # 5% chance of congestion
    congestion_magnitude = np.where(congestion_events, -20, 0)
    speed = base_speed + noise + congestion_magnitude
    speed = np.clip(speed, 10, 80)   # realistic bounds

    # Volume (vehicles per 5 min) correlated with speed (lower speed = higher volume)
    volume = 200 - 2.5 * speed + np.random.normal(0, 20, intervals)
    volume = np.clip(volume, 20, 400)

    # Weather data: temperature (C) and precipitation (mm/h)
    weather_temp = 20 + 5 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 3, intervals)
    weather_precip = np.random.exponential(0.5, intervals)
    weather_precip = np.where(np.random.rand(intervals) < 0.1, weather_precip, 0)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'speed': speed,
        'volume': volume,
        'weather_temp': weather_temp,
        'weather_precip': weather_precip
    })
    df['hour'] = hour
    df['day_of_week'] = day_of_week
    return df

if __name__ == '__main__':
    df = generate_traffic_data(days=60, sample_interval=5)
    df.to_csv('traffic_data.csv', index=False)
    print(f"Generated {len(df)} records -> traffic_data.csv")
