import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for the ReconGuard model.
    
    Expected input columns:
    - user_id
    - merchant_id
    - amount
    - timestamp (datetime)
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # 1. Time-of-day buckets
    # 0: Night (0-6), 1: Morning (6-12), 2: Afternoon (12-18), 3: Evening (18-24)
    df['hour'] = df['timestamp'].dt.hour
    df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False, include_lowest=True).astype(int)
    
    # 2. Merchant averages
    # Calculate historical average amount for each merchant (expanding mean to avoid leakage, or simple group mean if offline)
    # Using expanding mean to simulate real-time availability
    df['merchant_avg_amount'] = df.groupby('merchant_id')['amount'].transform(lambda x: x.expanding().mean())
    
    # 3. Rolling 24h user features
    # We need to set the index to timestamp for rolling windows
    df_indexed = df.set_index('timestamp')
    
    # Group by user and calculate rolling stats over 24h
    # count: number of transactions in last 24h
    # sum: total amount spent in last 24h
    # mean: average transaction amount in last 24h
    
    user_rolling = df_indexed.groupby('user_id')['amount'].rolling('24h')
    
    df['user_24h_count'] = user_rolling.count().values
    df['user_24h_sum'] = user_rolling.sum().values
    df['user_24h_mean'] = user_rolling.mean().values
    
    # Fill NaNs that might result from the first record or rolling windows
    df = df.fillna(0)
    
    return df
