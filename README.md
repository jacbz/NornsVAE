# NornsVAE

![](https://norns.community/community/jacbz/nornsvae.png)

NornsVAE is a machine-learning-powered drum sequencer for the [Norns](https://monome.org/norns/) platform. It uses Google's [MusicVAE](https://magenta.tensorflow.org/music-vae) model to generate drum sequences. Because of the limited computing power on a Rasberry Pi, computations are performed by a separate Python server, which needs to be independently run on a computer in the same network.

It is being developed as part of my Master's thesis in computer science, which aims to research interactive music generation with deep learning.

The project consists of three modules:

1. a client, written as a Norns script in Lua
2. a server, implemented as a Python server (Flask) that wraps a MusicVAE (Tensorflow) instance
3. a logging server (JavaScript / Express.js / MongoDB)

All actions performed on the client and server are sent to the logging server for evaluation as part of my thesis. Logs are anonymized and contain no identifiable user information. By using this software, you agree to logging.

## Running the server
Required: [Python 3](https://www.python.org/)

1. Download the server [here](https://github.com/jacbz/NornsVAE/releases) (this includes the machine learning model and other assets)
2. Install python dependencies using
	```
	pip install -r requirements.txt
	```
3. Run the server using
	```
	python server.py
	```
4. If successfully started, the console will display:
	```
	* Running on http://<ip address>:5000/ (Press CTRL+C to quit)
	```
    Remember this IP address.

### Troubleshooting
- If you get errors with `rtmidi` (a dependency), you may need to install headers for some sound libraries. On Ubuntu, this should work:
	```
	sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev
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