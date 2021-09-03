from train import SpeakerClassifier
import toml
import pathlib

if __name__ == '__main__':
    config = toml.load(
        pathlib.Path(r"/home/tharun/speaker-count/Tharun/config.toml"))

        
    classifier = SpeakerClassifier()
    classifier.train()
