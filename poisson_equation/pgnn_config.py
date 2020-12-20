# const
N_INPUTS = 2
N_OUTPUTS = 1

# supported values: 1000, 2000, 4000, 8000, 16000
BATCH_SIZE = 1000

# variables
N_LAYERS = 3
MAX_ITER = 1000
SEED = 41

N_UNITS = []
for i in range(N_LAYERS):
    N_UNITS.append(16)
