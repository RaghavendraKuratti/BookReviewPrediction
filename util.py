import pickle
from flask import Flask

def load_artifects(text):
	with open('./Models/vector.pkl', 'rb') as f:
		vector = pickle.load(f)
	test_data = vector.transform([text])
	with open('./Models/sentiment_classifier.pkl', 'rb') as f:
		loaded_pkl = pickle.load(f)
	return loaded_pkl.predict(test_data)[0]

# if __name__ == "__main__":
# 	load_artifects("IM Bad")