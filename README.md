# Emotion Recognition by Electroencephalogram(EEG)
EEG analysis using DEAP dataset
```python
## How to run the model
# Clone the repository
git clone https://github.com/konkuk2023/final-project.git

# Make a environment
conda env create -f requirements/vggish.yaml
conda activate vggish
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements/requirements.txt
pip install gspread
pip install --upgrade auth2client

# Train/Test the model (example)
sh scripts/valence_10s.sh
```