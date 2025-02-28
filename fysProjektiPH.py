import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Load location data
path_location = "Location.csv"
df_location = pd.read_csv(path_location)

# Load acceleration data
path_kiihtyvyys = "kiihtyvyys.csv"
df_kiihtyvyys = pd.read_csv(path_kiihtyvyys)

# Function to define bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the filter
def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

# Extract time and Y-axis acceleration
time = df_kiihtyvyys['Time (s)']
accel_y = df_kiihtyvyys['Y (m/s^2)']

# Compute sampling frequency
T = time.iloc[-1] - time.iloc[0]  # Total duration
n = len(time)  # Number of data points
fs = n / T  # Sampling frequency (Hz)

# Define bandpass filter parameters
lowcut = 0.3  # Lower bound of step frequency (Hz)
highcut = 3.0  # Upper bound of step frequency (Hz)

# Apply bandpass filter to the Y-axis acceleration
filtered_accel_y = apply_bandpass_filter(accel_y, lowcut, highcut, fs)

# Detect both peaks (upward motion) and troughs (downward motion)
peaks, _ = find_peaks(filtered_accel_y, height=0.5, distance=int(fs * 0.5))
troughs, _ = find_peaks(-filtered_accel_y, height=0.5, distance=int(fs * 0.5))  # Detect negative peaks

# Total steps = peaks + troughs
total_steps_filtered = len(peaks) + len(troughs)

# Plot the filtered acceleration signal with both peaks and troughs
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time, filtered_accel_y, label="Filtered Acceleration (Y-Axis)")
ax.plot(time.iloc[peaks], filtered_accel_y[peaks], "ro", label="Detected Steps (Peaks)")  # Mark positive peaks
ax.plot(time.iloc[troughs], filtered_accel_y[troughs], "bo", label="Detected Steps (Troughs)")  # Mark negative peaks
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")
ax.set_title("Filtered Acceleration Signal & Detected Steps")
ax.legend()
ax.grid(True)

from scipy.fftpack import fft, fftfreq

# ------------------------- Fourier Analysis -------------------------
# Compute Fourier transform on the filtered acceleration data
N = len(filtered_accel_y)
dt = time.iloc[1] - time.iloc[0]  # Time difference between samples
freq = np.fft.fftfreq(N, d=dt)
fft_values = np.fft.fft(filtered_accel_y)
psd = (np.abs(fft_values)**2) / N  # Power spectral density

# Only consider positive frequencies
pos_mask = freq > 0
freq_positive = freq[pos_mask]
psd_positive = psd[pos_mask]

# Restrict to the step frequency band (0.5 to 3.0 Hz)
band_mask = (freq_positive >= lowcut) & (freq_positive <= highcut)
freq_band = freq_positive[band_mask]
psd_band = psd_positive[band_mask]

# Determine the dominant frequency in the band
if len(psd_band) > 0:
    dominant_index = np.argmax(psd_band)
    dominant_frequency = freq_band[dominant_index]
else:
    dominant_frequency = 0

# Calculate Fourier-based steps (assuming 2 steps per cycle)
total_steps_fft = int(np.round(dominant_frequency * T * 2))




# Function to calculate total distance using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Total distance
df_location['prev_lat'] = df_location['Latitude (°)'].shift()
df_location['prev_lon'] = df_location['Longitude (°)'].shift()
df_location['distance_km'] = df_location.apply(lambda row: haversine(row['prev_lat'], row['prev_lon'], row['Latitude (°)'], row['Longitude (°)']), axis=1)

# Sum total distance
total_distance_km = df_location['distance_km'].sum()

#Print values and visualisations to browser
st.title('Testimatka')
st.write("Askelten lukumäärä (Fourier-menetelmällä laskettu):", total_steps_fft)
st.write("Askelten lukumäärä (suodatuksella havaittu) : ", total_steps_filtered)
st.write("Ero suodatetun ja Fourier-menetelmän askelmäärissä:", abs(total_steps_filtered - total_steps_fft))
st.write("Keskinopeus on : ", round(df_location['Velocity (m/s)'].mean(), 2), 'm/s')
st.write("Kokonaismatka on : ", round(total_distance_km, 2), 'km')
st.write("Askeleen keskimääräinen pituus (suodatettu kiihtyvyysdatan perusteella):", round(total_distance_km / total_steps_filtered * 1000, 2), 'metriä')
# Display the filtered acceleration plot
st.pyplot(fig)
# Displau the power spectrum plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(freq_band, psd_band, 'b-', marker='o')
ax.set_xlabel('Taajuus [Hz]')
ax.set_ylabel('Teho')
ax.set_title('Power Spectrum of Filtered Y-Acceleration')
ax.grid(True)

st.pyplot(fig)

#Create a map where the center is at the start_lat start_long and zoom level is defined
start_lat = df_location['Latitude (°)'].mean()
start_long = df_location['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_long], zoom_start=15)

#Draw the map
folium.PolyLine(df_location[['Latitude (°)','Longitude (°)']], color = 'blue', weight = 3.5, opacity =1).add_to(map)
#Define map dimensions and show the map
st_map = st_folium(map, width=900, height=650)



