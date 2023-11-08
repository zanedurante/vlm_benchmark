# Instructions for downloading each dataset

### TODO: MOMA, Kinetics-100, SMSM

### HMDB-51

* Download `hmdb51_org.rar` and `test_train_splits.rar` from `https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads`.
* Create a new folder (which will store uncompressed dataset), and place both downloaded files into it.
* In `dataset/dataset_handler.py`, change the constant `HMDB_51_DIR` to equal the path to your new folder.
* The dataset_handler script will automatically extract necessary files when the dataset is accessed.

### UCF-101

* Download `UCF101.rar` and `UCF101TrainTestSplits-RecognitionTask.zip` from `https://www.crcv.ucf.edu/data/UCF101.php`.
* Create a new folder (which will store uncompressed dataset), and place both downloaded files into it.
* In `dataset/dataset_handler.py`, change the constant `UCF_101_DIR` to equal the path to your new folder.
* The dataset_handler script will automatically extract necessary files when the dataset is accessed.

### Something-Something v2 (Original full dataset, as opposed to smsm, which uses 100 / 174 classes)

* Download instructions, videos and labels from `https://developer.qualcomm.com/software/ai-datasets/something-something`.
* Follow instructions to concatenate and extract videos from archive form into a single directory called `20bn-something-something-v2`.
* Create a new folder (to store both uncompressed vid folder and label info), then change the constant `SSV2_DIR` in `dataset/dataset_handler.py` to match the folder's path.
* Move the uncompressed video folder (`20bn-something-something-v2`) into the new folder.
* Unzip the label file `20bn-something-something-download-package-labels.zip`, and move the `labels` folder into the newly created dataset folder (alongside uncompressed video folder).