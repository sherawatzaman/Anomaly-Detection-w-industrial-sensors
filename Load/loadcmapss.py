{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "cols = [\"engine\", \"cycle\"] + [f\"setting{i}\" for i in range(1, 4)] + [f\"s{i}\" for i in range(1, 22)]\n",
    "train = pd.read_csv(\"../data/cmapss/train_FD001.txt\", sep=r\"\\s+\", header=None, names=cols)\n",
    "\n",
    "print(train.shape)   # expect (20631, 26)\n",
    "train.head()\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
