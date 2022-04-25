CLASS_NAMES = [
    'OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak',
    'Flower'
]

VIS_ALL_LABELS = ['OK', 'SK', 'DEAD', 'EM', 'NORM', 'ST', 'WK', 'FR']

VIS_BINARY_LABELS = ['OK', 'NO_OK']

BN_CLASS_NAMES = ['OK', 'NoOK']

HEADER_NAMES = ['filename'] + CLASS_NAMES

THRESH = [0.9, 0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

BASE_HEADERS = ['Fold', 'Rank', 'Epoch', 'Step']

MODEL_SELECTION_RECORD_HEADERS = BASE_HEADERS + [
    'Loss', 'ROC_AUC', 'Other_Loss', 'OTHER_ROC_AUC'
]

PERFORMANCE_RECORD_HEADERS = BASE_HEADERS + [
    'class', 'precision', 'in_precision', 'recall', 'f1-score', 'support'
]