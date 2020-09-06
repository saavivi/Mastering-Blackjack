# Mastering-Blackjack: Combining Reinforcement Learning with Deep Reinforcement Learning

<img src="GitHub.gif" width="250" height="250"/>

- [Mastering-Blackjack: Combining Reinforcement Learning with Deep Reinforcement Learning](#Mastering-Blackjack: Combining Reinforcement Learning with Deep Reinforcement Learning)
  * [Running The Project](#running-the-project)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)

## Running The Project
|Step      | Description |
|-------------|---------|
|1| Clone the project to your local computer |
|2| Install required packages in requiremnts.txt |
|3| Run run_experiments.py |
|4| Output files will apper under a new experiments folder |


### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/saavivi/Mastering-Blackjack.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)


## Installation Instructions

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/distribution/
2. Create a new environment for the Project:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name torch`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate torch`
4. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`pytorch` (cpu)| `conda install pytorch torchvision cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch` |



