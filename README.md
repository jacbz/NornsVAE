# NornsVAE
![](https://github.com/jacbz/NornsVAE/workflows/build_server/badge.svg)

![](https://norns.community/community/jacbz/nornsvae.png)

NornsVAE is a machine-learning-powered drum sequencer for the [Norns](https://monome.org/norns/) platform. It uses Google's [MusicVAE](https://magenta.tensorflow.org/music-vae) model to generate drum sequences. Because of the limited computing power on a Rasberry Pi, computations are performed by a separate Python server, which needs to be independently run on a computer in the same network.

**[Introduction video on YouTube](https://youtu.be/sj_bG7nzDqU)**

It is being developed as part of my Master's thesis in computer science, which aims to research interactive music generation with deep learning.

The project consists of three modules:

1. a client, written as a Norns script in Lua
2. a server, implemented as a Python server (Flask) that wraps a MusicVAE (Tensorflow) instance
3. a logging server (JavaScript / Express.js / MongoDB)

All actions performed on the client and server are sent to the logging server for evaluation as part of my thesis. By using this software, you agree to logging. When the user study is completed, logging will be removed.

For more information on the user study, see the post on [lines.co](https://llllllll.co/t/nornsvae-ml-powered-drum-sequencer-for-norns/55624).

## Running the server
There are two options for running server:
1. Pre-built distributable (Windows)
2. Running from source (Mac)

If successfully started, the console will display:
```
* Running on http://<ip address>:5000/ (Press CTRL+C to quit)
```

### Pre-built distributable (Windows)
Download the distributables [here](https://github.com/jacbz/NornsVAE/releases/download/release/nornsvae_server_windows.zip) and run `server.exe`. The distributable is quite large because of package dependencies (in particular, Tensorflow).

### Running from source (Mac)
On Mac, you'll need to run the server from source. You'll need Python and Miniforge.

#### Installation
1. Download [Miniforge](https://github.com/conda-forge/miniforge).

	If you have an M1 Mac, you'll need the arm64 (Apple Silicon) version. Download the script [here](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) and install it:
	```
	bash Miniforge3-MacOSX-arm64.sh
	```
	
	If you have an Intel Mac, you'll need the x86_64 version. Download the script [here](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh) and install it:
	```
	bash Miniforge3-MacOSX-arm64.sh
	```

2. Once this is complete, you can set up a new Conda environment where we will install the required packages:
	```
	conda create --name nornsvae python=3.8
	```
	Now you can use (activate) this environment:
	```
	conda activate nornsvae
	```
	The command line should now display `(nornsvae)`.
3. Install Tensorflow inside the new environment:
	```
	conda install -c apple tensorflow-deps
	```
	If you have an M1 Mac, install `tensorflow-macos`:
	```
	pip install tensorflow-macos
	```
	If you have an Intel Mac, install `tensorflow`:
	```
	pip install tensorflow
	```
4. Download the NornsVAE Server [here](https://github.com/jacbz/NornsVAE/releases/download/release/nornsvae_server_source.zip) and unzip it. Run the following commands from within the `server` folder:
	```
	pip install -r requirements.txt
	```
	Make sure that you are in the NornsVAE environment: `conda activate nornsvae`
	
If you have trouble installing the requirements, see the [Troubleshooting Guide](TROUBLESHOOTING.md).

#### Run
Once you have installed all dependencies, simply navigate to the `server` folder and run the server using `python server.py`. Make sure that you are in the NornsVAE environment: `conda activate nornsvae`

The terminal will display the local IP address of your computer, if everything works.

## Running the client
1. From maiden, type
	```
	;install https://github.com/jacbz/nornsvae_client
	```
2. Write the IP address of the server into the file `server-ip`, located inside the script folder
3. You can now start using the script!

## Links
- View on [norns community](https://norns.community/en/authors/jacbz/nornsvae)