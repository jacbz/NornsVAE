# NornsVAE
![](https://github.com/jacbz/NornsVAE/workflows/build_server/badge.svg)

![](https://norns.community/community/jacbz/nornsvae.png)

NornsVAE is a machine-learning-powered drum sequencer for the [Norns](https://monome.org/norns/) platform. It uses Google's [MusicVAE](https://magenta.tensorflow.org/music-vae) model to generate drum sequences. Because of the limited computing power on a Rasberry Pi, computations are performed by a separate Python server, which needs to be independently run on a computer in the same network.

It is being developed as part of my Master's thesis in computer science, which aims to research interactive music generation with deep learning.

The project consists of three modules:

1. a client, written as a Norns script in Lua
2. a server, implemented as a Python server (Flask) that wraps a MusicVAE (Tensorflow) instance
3. a logging server (JavaScript / Express.js / MongoDB)

All actions performed on the client and server are sent to the logging server for evaluation as part of my thesis. By using this software, you agree to logging. When the user study is completed, logging will be removed.

## Running the server
There are two options for running server:
1. Pre-built distributable (Windows)
2. Running from source (Mac)

If successfully started, the console will display:
```
* Running on http://<ip address>:5000/ (Press CTRL+C to quit)
```

### Pre-built distributable (Windows)
Download the distributables [here](https://github.com/jacbz/NornsVAE/releases) and run `server.exe`. The distributable is quite large because of package dependencies (in particular, Tensorflow).

### Running from source (Mac)
On Mac, you'll need to run the server from source. You'll need Python and Miniforge.

1. Download Miniforge [here](https://github.com/conda-forge/miniforge). If you have an M1 Mac, you'll need the Apple Silicon version. Run the downloaded install script:
	```
	bash Miniforge3-MacOSX-arm64.sh
	```
2. Once this is complete, you can set up a new Conda environment where we will install the required packages:
	```
	conda create --name nornsvae python=3.8
	conda activate nornsvae
	```
3. Install Tensorflow inside the new environment:
	```
	conda install -c apple tensorflow-deps
	pip install tensorflow-macos
	```
4. Navigate to the `server` folder inside the NornsVAE repo, and install the other dependencies:
	```
	pip install -r requirements.txt
	```
5. Download the assets [here](https://home.in.tum.de/~zhangja/nornsvae/assets.zip) (this includes the machine learning model) and copy the `assets` folder into the `server` folder
6. Run the server using
	```
	python server.py
	```
When you run the server, make sure that you are using the `nornsvae` environment (`conda activate nornsvae`).

## Running the client
1. From maiden, type
	```
	;install https://github.com/jacbz/nornsvae_client
	```
2. Write the IP address of the server into the file `server-ip`, located inside the script folder
3. You can now start using the script!

## Links
- View on [norns community](https://norns.community/en/authors/jacbz/nornsvae)