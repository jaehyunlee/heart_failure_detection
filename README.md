# Heart Failure Detection
This is code for detecting subjects with congestive heart failure(NYHA classes I, II, and III) and normal
* Based on CNN and RR interval data
* The beat annotations were digitized at 128 samples per second

## Version
* Python (3.6.3)
* tensorflow (1.4.1)
* tensorflow-tensorboard (0.4.0rc3)
* numpy (1.13.3)

## Training Data
This code makes two files:
* check point file(.ckpt) for saving train result
* log file for using tensorboard
```bash
python rr_train.py
```

## Predict Data
This code restores train result and checks accuracy
```bash
python rr_predict.py
```

## Reference Data
* https://www.physionet.org/

## Reference Implementations
* https://github.com/hunkim/DeepLearningZeroToAll/
* https://github.com/bwcho75/tensorflowML/
* https://github.com/bwcho75/facerecognition/
