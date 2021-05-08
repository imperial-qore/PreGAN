# Directory paths
model_folder = 'recovery/PreGANSrc/checkpoints/'
data_folder = 'recovery/PreGANSrc/data/'
plot_folder = 'recovery/PreGANSrc/plots'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 5
PERCENTILES = 99
PROTO_DIM = 2
PROTO_UPDATE_FACTOR = 0.2
PROTO_UPDATE_MIN = 0.05
PROTO_FACTOR_DECAY = 0.999

# GAN parameters
Coeff_Energy = 0.8
Coeff_Latency = 0.2