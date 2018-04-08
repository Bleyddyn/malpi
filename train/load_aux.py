import os
import sys
import json
import numpy as np
from keras.utils import to_categorical

def sameAux( aux1, aux2 ):
    for idx in ["name", "type", "categories"]:
        if aux1[idx] != aux2[idx]:
            return False
    return True

def embedAux( actions, aux ):
    embedding = {}
    for idx, act in enumerate(aux["categories"]):
        embedding[act] = idx
    emb = []
    for act in actions:
        emb.append( embedding[act] )

    return emb

def loadOneAux( drive_dir, aux ):

    auxFile = os.path.join( drive_dir, "{}_aux.npy".format(aux["name"] ) )
    if os.path.exists(auxFile):
        actions = np.load(auxFile)
        actions = actions.astype('str')
    else:
        print( "Missing auxiliary data file: {}".format( auxFile ) )
        actions = []

    return actions

def loadAuxData( dirs, auxName ):

    actions = []
    aux = None
    count = 1

    for onedir in dirs:
        if len(onedir) > 0:
            aux_file = os.path.join( onedir, "aux.json" )
            if os.path.exists(aux_file):
                with open(aux_file,'r') as f:
                    auxMeta = json.load(f)
                if auxName in auxMeta:
                    nc = auxMeta[auxName]
                    if aux is None:
                        aux = nc
                    elif not sameAux( aux, nc ):
                        print( "Error: Inconsistent auxiliary meta data:\n{}\n{}".format(aux,nc) )
                        return None
                    dactions = loadOneAux( onedir, aux )
                    try:
                        tmpy = embedAux( dactions, aux )
                    except:
                        print( "embed failed: {}".format( onedir ) )
                    actions.extend(dactions)
                    print( "Loading {} of {}: {} auxiliary data".format( count, len(dirs), len(actions) ), end='\r' )
                    sys.stdout.flush()
                    count += 1

    print("")

    y = embedAux( actions, aux )
    y = to_categorical( y, num_classes=len(aux["categories"]) )
    return y

def badAuxData( dirs, auxName ):
#{"LanePosition": {"name": "LanePosition", "type": "categorical", "default": "CenterLine", "categories": ["OutsideTrack", "OutsideLine", "LeftLane", "CenterLine", "RightLane", "InsideLine", "InsideTrack"]}}

    count = 1

    for onedir in dirs:
        if len(onedir) == 0:
            continue

        aux_file = os.path.join( onedir, "aux.json" )
        if not os.path.exists(aux_file):
            print( "Missing aux file: {}".format( aux_file ) )
            continue

        with open(aux_file,'r') as f:
            auxMeta = json.load(f)
        if auxName in auxMeta:
            print( "Checking {} of {}: {}".format( count, len(dirs), onedir ) )
            count += 1

            aux = auxMeta[auxName]
            embedding = {}
            for idx, act in enumerate(aux["categories"]):
                embedding[act] = idx
            dactions = loadOneAux( onedir, aux )
            data_str = []
            for act in dactions:
                data_str.append(str(act))

            good = True
            emb = []
            for idx, act in enumerate(data_str):
                try:
                    emb.append( embedding[act] )
                except:
                    print( "  bad data at {}: {}".format( idx, act ) )

    print("")

def getDirs(fname):
    with open(fname, "r") as f:
        dirs = f.read().split('\n')

    for i in reversed(range(len(dirs))):
        if dirs[i].startswith("#"):
            del dirs[i]
        elif len(dirs[i]) == 0:
            del dirs[i]

    return dirs

if __name__ == "__main__":
    dirs = getDirs("dirs.txt")
    aux = "TestAux"

    #auxData = loadAuxData( dirs, aux )
    badAuxData( dirs, aux )
