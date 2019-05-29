## Installation
### Prerequisites
- Windows 7 / 10

- StarCraft: Brood War I (1.16.1)

    Install StarCraft: Brood War, and update it to version patch 1.16.1.(not support 1.18.1)
    Since Blizzard have taken off v1.16.1 and earlier from their website, ICCUP has some instructions on how to install an older version of [BroodWar that is compatible with BWAPI (needs the v1.16.1 patch)](http://iccup.com/en/starcraft/sc_start.html)

    Make installation path to __C:\Starcraft__

- Python

    To get started, you’ll need to have Python 3.6 x64 installed with keras and tensorflow as backend and protobuf and other dependencies.


First of all, you can clone this repository from github.
```
git clone https://github.com/TeamSAIDA/SAIDA_RL.git
```

For your comfort, you can install all required dependencies with a single command below.

```
cd SAIDA_RL\python
pip install -r requirements.txt
```

And add __path\to\SAIDA_RL\python__ to __PYTHONPATH__ of environment variable like below.


After that, run batch file "install\copyFiles.cmd" to copy usemap and SAIDA RL execute files to StarCraft folder.
```
install\copyFiles.cmd
```

And you’re good to go!
