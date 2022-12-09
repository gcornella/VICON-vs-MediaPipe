# VICON-vs-MediaPipe

**Abstract—** This project aims to compare the performance of two different camera-based approaches to analyze lowerbody joint degrees. The comparison is between the [VICON](https://www.vicon.com/about-us/what-is-motion-capture/) motion-capture system and the computer vision-based [MediaPipe](https://github.com/google/mediapipe) framework implemented in Python. A GUI has been designed to provide physically impaired individuals with rehabilitation exercises that can be performed from home and exclusively with a smartphone.

## Installation
Download the folder to your preferred directory,
```bash
git clone https://github.com/gcornella/VICON-vs-MediaPipe.git
```
and install the requirements file using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

## Execution
Run the main file inside the program folder.
```python
python main.py
```
A GUI will appear automatically, and you will have to choose some inputs that will modify the functionalities of the experiment.
The GUI is designed with [tkinter](https://docs.python.org/3/library/tkinter.html), as it can be seen in the file input_selection.py

```python
exercising         # If True, the GUI proposes new knee degrees every time the user pressed the key 'p'
plot_exercise_line # If True, the exercise line is going to be plotted on top of the screen
save               # If True, the program saves an excel file and a .mat file when it finishes executing
udp_open           # If True, the UDP communcation is allowed, and then you just have to run the plotter
```

If you want to plot the knee degrees in a real-time external plotter, allow the UDP communication protocol and run the following file in another session.

```python
python plotter.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

If you have any suggestions or would like to know more about the experiment, please read the attached .pdf report, and feel free to contact me for further details.

*This project has been developed by members of the BioRobotics Lab @luisgarcia and @guillemcornella, both Ph.D. students at the University of California Irvine.*
