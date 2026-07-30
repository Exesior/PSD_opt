"""Test NaN behavior in mean calculation."""

import numpy as np

# Test 1: Mean with NaN
arr = np.array([0.6, 0.6, np.nan, 0.6])
print(f"Array with NaN: {arr}")
print(f"np.mean(): {np.mean(arr)}")
print(f"np.nanmean(): {np.nanmean(arr)}")

# Test 2: Masking
valid_mask = ~np.isnan(arr)
print(f"\nValid mask: {valid_mask}")
print(f"Mean of valid: {np.mean(arr[valid_mask])}")

# Test 3: All NaN
arr_all_nan = np.array([np.nan, np.nan, np.nan])
print(f"\nAll NaN array mean: {np.mean(arr_all_nan)}")
print(f"All NaN array nanmean: {np.nanmean(arr_all_nan)}")

# Test 4: Empty after masking
arr_mixed = np.array([np.nan, np.nan])
valid = ~np.isnan(arr_mixed)
print(f"\nEmpty after masking:")
print(f"  arr[valid]: {arr_mixed[valid]}")
print(f"  np.mean(arr[valid]): {np.mean(arr_mixed[valid]) if len(arr_mixed[valid]) > 0 else 'EMPTY'}")
