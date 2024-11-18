# Software-based test images for in-field fault detection of hardware accelerators
This repositoy works as a utility library to successfully generate and validate the ITL (Image Test Library) technique for ultra-low power Edge Accelerators (i.e., the RISC-v 32 bit integer multiplier used by the X-HEEP platform).
Note: to date [10/2024] an interpreter < "Python 3.12" must be used due to the lack of compatibility between Tensoflow Lite module and the latest version of the interpreter.

Calling the main function, different functionalities can be chosen to be executed:
 - `python3 main.py --save_weights` -- save to file the trained weights of the specified model and layer in the specified format (hex, bin, int8)
 - `python3 main.py --save_pairs` -- save to file the <input, weight> pairs involved during the convolution algorithm of the specified model and convolutional layer
 - `python3 main.py --generate_atpg_scripts` -- generate .tcl scripts for the ATPG process, one .tcl for each trained signed weight of the first convolutinal layer
 - `python3 main.py --save_model_hex_format` -- save the model in .hex (words/bytes) to run it into the X-HEEP platform running on the PYNQ-Z2 board
 - `python3 main.py --gather_patterns_input_positions` -- collect, for all the test patterns, all the input positions which are multiplied for a given a weight which is, in turn, multiplied for a given test pattern
 - `python3 main.py --generate_FI_files` -- collect itl-validation fault injection files
 - `python3 main.py --generate_custom_input_image` -- generate a input image as specified by this parameter (e.g. FWP = fill with pattern)
  - `python3 main.py --print_image` -- print a .png version of an input image stored as .npy tensor

The .ipynb file contained in this repository allows the complete management of the LeNet-5 network, and input image creation exploiting the IICV technique.

# Contacts
Giacomo Perlo:  [Linkedin](https://www.linkedin.com/in/giacomo-perlo/), <perlogiacomo@gmail.com> <br/>
If you need help or want to know something more about all of this, I'm ready and excited to help you!

# Licence
[LICENCE](LICENCE)

# Presentation
![Example Image](https://drive.google.com/uc?export=view&id=1gmyyAsoEoneRPAPkVl0HfN0RozlomRnX)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1txYRPwq118y1TS67hz3yaBj849TeSwFN)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=17wm58kk3Jf0vFXSU3ZDJeD75n32W1rwQ)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1NVsI18ZjSeNZNj0rlLah0IhcjHlth4kf)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1DPnpMRB1Q82E03noT8JF5f4EdX1gw310)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=11IhGnxPHRYtdpeCJheWXz-6SVORja-pr)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1EaISePofMPbrhB1fZAgfKjNy830lXhjb)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=18X87bFBoweyiw72l06HDdX7uBtSAfFg2)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1gdBPq_yJPgCkvO9t5Y-tArL9-29fhKJW)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1Ck4WoRlsJcEc4rHmok_Nk30atQ_W8Ndv)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1j0Z4Fseyf-9_ZlzaUDqepBe-N8PFiaoW)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1ofmZgUGyxPobRLYolVCO5kGmywiJ9KUw)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1BgXYGQXK4Jc90bz5jLLsRBpuTs2n8710)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1XSZvS4EKggrkByNGNn5bAn1QJGhYsb1X)<br/>
![Example Image](https://drive.google.com/uc?export=view&id=1PNKDO_EZtuwrqf3c1E6CLFgMRjO_IWKa)<br/>

