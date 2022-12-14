# NTHU CCXP Decaptcha

Automatically fill in the captcha on the NTHU Academic Information Systems.

## Download & Installation

- [Firefox Add-on](https://addons.mozilla.org/zh-TW/firefox/addon/nthu-ccxp-decaptcha/)
- [Chrome Extension](https://chrome.google.com/webstore/detail/nthu-ccxp-decaptcha/hpbhebpkmhpeoomcmdmlmhlclhbbdjho?hl=zh-TW)

## Privacy Policy

This extension does not collect any personal data.

## Implementation Details

### Data Collection

- Download and manually label the images

### Data Preprocessing

- Split the image into six patches of equal width

### Model

- Conv2d Layer x 2
- BatchNorm Layer after each Conv2d
- ReLU as the activation function
- Apply Adaptive Average Pooling to get 2x2 feature
- Linear layer x 1

### Extension

- Download the captcha image
- Transform to an Array of bytes
- JSON.stringify the array and pass it to the API
