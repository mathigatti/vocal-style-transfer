n_epochs = 2000 
g_type = "gated_cnn"  # or "u_net"
began = False  # True : Cycle-BeGan, False : CycleGan

sr = 16000  # sampling rate
n_features = 24 # Mceps coefficient 
n_frames = 128   # fixed-length segment randomly 
frame_period = 5.0 # extracted every 5 ms

dataset_A = "./data/train/A"
dataset_B = "./data/train/B"
test_dir = "./data/test"
direction = "A2B"

log_dir = "./log"
model_dir = "./model"