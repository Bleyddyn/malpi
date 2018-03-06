""" DriveFormat.
    A base class for file formats meant to be used by dagger.py.

    TODO:
    1) Have a method that returns a list of action space names, types and sizes.
        e.g. [{ "name": "steering", "type": "categorical", "size": 5},
              { "name": "throttle", "type": "continuous", "size": (-1.0,1.)}]
    2) Add support for auxiliary labels as defined by the user. Name, type, size.
        e.g. Let the user label data based on which lane the car is in.
"""

class DriveFormat:
    """ A class to represent a drive on disc.
    """

    _formats = {}

    def __init__( self ):
        self.clean = True

    @staticmethod
    def registerFormat( name, formatClass ):
        """ This method should be called by sub-classes after they are defined.
        This method should NOT be overridden by sub-classes."""
        if name in DriveFormat._formats:
            if DriveFormat._formats[name] == formatClass:
                return
            else:
                raise ValueError("Multiple Drive formats with the same name: " + str(name) )
        DriveFormat._formats[name] = formatClass

    @staticmethod
    def classForFile( path ):
        """ This method will be called by the UI to find the appropriate class for a given file.
        Sub-classes must implement canOpenFile(path) as a classmethod."""
        for name, cls in DriveFormat._formats.items():
            if cls.canOpenFile(path):
                return cls
        return None

    @classmethod
    def canOpenFile( cls, path ):
        """ Sub-classes must override this as a class method.
        Return is True if the class can read and write the file at path. Otherwise False."""
        return False
                    
    def save( self ):
        """ Override this to save the file and either call this or call setClean() to
        mark it as clean """
        self.clean = True

    def setDirty( self ):
        """ Mark this file as dirty, i.e. it has been edited since the last time it was
        saved """
        self.clean = False

    def setClean( self ):
        """ Mark this file as clean, i.e. no edits since last saved. """
        self.clean = True

    def isClean( self ):
        """ Returns True if this file has not be edited since opened or last saved.
        False, otherwise. """

        return self.clean

    def count( self ):
        """ Must be overridden by subclasses. Should return the number of samples
        in the file. """

        return 0

    def imageForIndex( self, index ):
        """ Return the image for the sample at index.
        The only image format currentlly supported is a numpy array with
        shape: ( height, width, 3 channels).
        e.g. return self.images[index]"""

        return None

    def actionForIndex( self, index ):
        """ Return the action for the sample at index.
        Action should be a string for categorical outputs, or a float for continuous outputs.
        e.g. return self.actions[index]"""

        return None

    def setActionForIndex( self, new_action, index ):
        """ Set the action for this index to new_action, then either call this method
        or call setDirty() directly to mark this file as edited.
        e.g. self.actions[index] = new_action; self.setDirty()"""

        self.setDirty()

    def actionNames(self):
        """ Return a list with strings representing each possible action.
        For continuous action spaces return an empty list.
        e.g. return [ 'forward', 'backward', etc... ]"""

        return [ ]

    def actionForKey(self,keybind,oldAction=None):
        """ Implement keybindings for this file type. The keybind argument will be a string
        with a single character the user typed. oldAction is for reference in case keybindings
        shift actions rather than choosing one.
        e.g. if keybind == 'w' then return 'forward'
        e.g. if keybind == '+' then return oldAction + 1"""

        return None

    def actionStats(self):
        """ Return a dictionary with action names as keys and a count of each action as value.
        TODO: Behavior isn't yet defined for continuous action spaces."""
        return {}

