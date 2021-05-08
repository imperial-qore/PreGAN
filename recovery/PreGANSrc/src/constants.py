# Directory paths
model_folder = 'recovery/PreGANSrc/checkpoints/'
data_folder = 'recovery/PreGANSrc/data/'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 5
PERCENTILES = 95
PROTO_DIM = 2
PROTO_UPDATE_FACTOR = 0.5
PROTO_UPDATE_MIN = 0.2
PROTO_FACTOR_DECAY = 0.999