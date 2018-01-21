import pickle

with open('drive.pickle','rb') as f:
    data = pickle.load(f)

print( data['model'] )
