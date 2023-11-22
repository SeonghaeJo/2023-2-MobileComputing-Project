# Image Description for Dementia Prevention
Project of 'Mobile Computing and Its Applications' in SNU

## Prepare tensorflow lite model
Contents of `image_captioning_custom.ipynb`
- Prepare Flickr8k dataset
- Load and quantize Resnet50 (Encoder)
- Train 2-layer transformer (Decoder) and save its weights into checkpoint
- Evaluate and compare two captioning models (original encoder vs quantized encoder)
- Other things (save vocabulary file, plot attention map...)

## How to run server for decoder model inference
1. Move into decoder_server directory.
In decoder_server directory, there is a `dockerfile` to install the dependencies like Tensorflow and Flask.
`controller.py` is the entrypoint of HTTP server.
`decoder.py` defines the decoder model and inference functions.

2. Execute the below commands. (Image for only AMD64 arch)
```shell
docker build -t tensorflow-decoder-server .
docker run -it -p 8123:8123 --name decoder_server tensorflow-decoder-server
```

3. Then you might be in the container. Move into /home, and run the below command.
```shell
python controller.py
```
