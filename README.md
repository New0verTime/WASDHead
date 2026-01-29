# WASDHead
## Introduction
Code for WASDHead: Low-Context-Switch Cursor Control with Head Mouse and Keyboard (IUI 2026 demonstration)
## Requirements
- Python 3.12
- Webcam
## Installation
### Clone the repository
``` git clone https://github.com/New0verTime/WASDHead.git ```
### Create and activate a virtual environment (optional but recommended)
``` python -m venv venv ```
```venv\Scripts\activate ```
### Install the required packages: ###
``` pip install -r requirements.txt ```
## Usage
### Run the application:
``` python app.py ```
### Adjust mouse parameter
- Firstly, set beta to 0 and mincutoff to a reasonable value such as 1.0
- Move head steadily at a very low speed to adjust mincutoff (decreasing mincutoff reduces jitter but increases lag)
- Secondly, move head quickly and increase beta until lag is minimized
- Note that, if high speed lag occurs, increase beta, if slow speed jitter appears, decrease mincutoff.
- Then, set a appropriate mouse speed to match your preference and comfort level
### Add preferred blendshapes bindings
- Add preferred blendshape bindings for mode switch and action.
- Test different facial expressions to find comfortable triggers
- Adjust sensitivity thresholds as needed
