# SimCSE
This is a PyTorch implementation of SimCSE

# Requirements
Then run the following script to install the remaining dependencies

```bash
pip install -r requirements.txt
```

# Dataset
We followed the official [SimCSE]() for the training dataset. Please download the wiki dataset for unsupervised SimCSE using `data/download_wiki.sh` and download the nli dataset for supervised SimCSE by running `data/download_nli.sh`.

# Train the model
To train the SimCSE,  simply run:

```bash
python run.py
```

# Future work
Add evaluation



