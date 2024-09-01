class config():
    
    NAME = "CNN_model"

    N_VARS = 2
    N_VARS_OUT = 1
    VARS_NAME_IN = []
    VARS_NAME_OUT = []
    NORMALISE_INPUT = False
    SCALE_OUTPUT = False
    FLUCTUATIONS_PREDICT = False
    RELU_THRESHOLD = -1.0
    TRAIN_YP = 0
    TARGET_YP = 50

    ON_GPU = True
    N_GPU = 1
    WHICH_GPU_TRAIN = 0
    WHICH_GPU_TEST = 1

    N_DATASETS = []
    N_SAMPLES_TRAIN = (1400, 1400, 1400, 1400, 1400, 1400, \
                       1400, 1400, 1400, 1400, 1400, 1400, \
                       1400, 1400, 1400, 1400, 1400, 1400 ) 
    INTERV_TRAIN = 3
    #N_SAMPLES_TEST = list(10)
    INTERV_TEST = 1
    NET_MODEL = 1
    #PAD_1 = WRAP

    N_EPOCH = 100
    BATCH_SIZE = 16
    VAL_SPLIT = 0.2

    INIT = 'random'
    INIT_MODEL = ""

    LR_INIT = 1e-3
    LR_DROP = 0.5
    LR_EPDROP = 40.0

    DATA_AUG = False


