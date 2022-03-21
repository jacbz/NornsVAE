# NornsVAE
![](https://github.com/jacbz/NornsVAE/workflows/build_server/badge.svg)

![](https://norns.community/community/jacbz/nornsvae.png)

NornsVAE is a machine-learning-powered drum sequencer for the [Norns](https://monome.org/norns/) platform. It uses Google's [MusicVAE](https://magenta.tensorflow.org/music-vae) model to generate drum sequences. Because of the limited computing power on a Rasberry Pi, computations are performed by a separate Python server, which needs to be independently run on a computer in the same network.

It is being developed as part of my Master's thesis in computer science, which aims to research interactive music generation with deep learning.

The project consists of three modules:

1. a client, written as a Norns script in Lua
2. a server, implemented as a Python server (Flask) that wraps a MusicVAE (Tensorflow) instance
3. a logging server (JavaScript / Express.js / MongoDB)

All actions performed on the client and server are sent to the logging server for evaluation as part of my thesis. Logs are anonymized and contain no identifiable user information. By using this software, you agree to logging.

## Running the server
There are two options for running server:
1. Pre-built distributable for Windows (.exe) and Mac
2. Running from source

If successfully started, the console will display:
```
* Running on http://<ip address>:5000/ (Press CTRL+C to quit)
```

### Pre-built distributable
Download the distributables [here](https://github.com/jacbz/NornsVAE/releases) and run `server.exe` (Windows) or `server` (Mac). The distributable is quite large because of package dependencies (in particular, Tensorflow).

### Running from source
If you wish to run the server from source, you will need Python (and preferably Anaconda).

1. Download the assets [here](https://home.in.tum.de/~zhangja/nornsvae/assets.zip) (this includes the machine learning model) and copy the `assets` folder into the `server` folder
2. Install python dependencies using
	```
	pip install -r requirements.txt
	```
3. Run the server using
	```
	python server.py
	```

## Running the client
1. From maiden, type
	```
	;install https://github.com/jacbz/nornsvae_client
	```
2. Write the IP address of the server into the file `server-ip`, located inside the script folder
3. You can now start using the script!

## Links
- View on [norns community](https://norns.community/en/authors/jacbz/nornsvae)