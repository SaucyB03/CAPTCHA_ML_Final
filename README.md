# CAPTCHA_ML_Final

## Instructions to run code:
- Code can be run from the python notebook titled "captcha.ipynb"
- Cells provide output representing parts of the process.
- In testing portion of the code, relevant parameters are set at the top of each cell
   - change them as needed to achieve reasonable runtimes.
- Code is written to be run locally, meaning the relevant python packages must be installed
   - no packages are used that have not been used previously throughout the course of the semester


## Process:
- Remove background gradient
- Isolate each character by slicing the Captcha images
- Convert images to black and white pixels
   - achieved via thresholding, any pixel above threshold is white, and all below are black
- Then, images can be passed to machine learning models for training and testing