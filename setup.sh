#!/bin/bash

echo "🔧 Setting up Patent Dashboard environment (macOS)..."

#Install Homebrew if missing
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "Homebrew already installed."
fi

#Install libomp (needed for xgboost on mac)
echo "Installing libomp..."
brew install libomp

#Create conda env
echo "Creating Conda environment: patent"
conda create -n patent python=3.10 -y

# activate env
eval "$(conda shell.zsh hook)"
conda activate patent

#Install python deps
echo "Installing Python packages..."
pip install streamlit pandas joblib matplotlib plotly xgboost==1.7.6

echo "Setup complete"
echo "Run your app with:"
echo "   conda activate patent"
echo "   streamlit run app.py"

#Make it executable in terminal with:
    #chmod +x setup.sh
    #./setup.sh
