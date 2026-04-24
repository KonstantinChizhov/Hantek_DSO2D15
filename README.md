
### Hantek DSO2D15 Oscilloscope automation python scripts

Creted with QWEN code.
Tested OS: Windows

##Get started
1. Connect the scope
2. install libusbK driver using Zadig tool
3. pip install pyvisa-py
4. Run the simple_test.py to verify

Files:
- diag_waveform_transfer.py - test reading waveform
- dso2d15.py - DSO2D15 Oscilloscope
- freq_response.py - continuous sin wave frequency response script 
- freq_response_burst.py - burst sin wave frequency response script 
- simple_test.py - simple read waveform and plot
