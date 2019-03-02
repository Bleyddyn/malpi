""" DriveFormat.
    A base class for file formats meant to be used by dagger.py.

    TODO:
    2) Add support for auxiliary labels as defined by the user. Name, type, names/range.
        e.g. Let the user label data based on which lane the car is in.
"""

class DriveFormat:
    """ A base class to represent a drive on disc.
    Callers can call handlerForFile(path) to get an object that can read/write the given file.
    Sub classes need to call registerFormat(...) so they will be known to DriveFormat.
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
    def handlerForFile( path ):
        """ This method will be called by the UI to find the appropriate class for a given file.
        Sub-classes must implement canOpenFile(path) as a classmethod."""
        for name, cls in DriveFormat._formats.items():
            if cls.canOpenFile(path):
                handler = cls(path)
                return handler
        return None

    @classmethod
    def canOpenFile( cls, path ):
        """ Sub-classes must override this as a class method.
        Return is True if the class can read and write the file at path. Otherwise False."""
        return False
                    
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

    def load( self, progress=None ):
        """ Override this to load the file and either call this or call setClean() to
            mark it as clean.
            progress should be a function taking two ints: samples read, total number of samples.
        """
        self.clean = True

    def save( self ):
        """ Override this to save the file and either call this or call setClean() to
        mark it as clean """
        self.clean = True

    def count( self ):
        """ Must be overridden by subclasses. Should return the number of samples
        in the file. """

        return 0

    def __len__(self):
        return self.count()

    def imageForIndex( self, index ):
        """ Must be overridden by subclasses.
        Return the image for the sample at index.
        The only image format currentlly supported is a numpy array with
        shape: ( height, width, 3 channels).
        e.g. return self.images[index]"""

        return None

    def actionForIndex( self, index ):
        """ Must be overridden by subclasses.
        Return the action for the sample at index.
        Action should be a string for categorical outputs, or a float for continuous outputs.
        e.g. return self.actions[index]"""

        return None

    def setActionForIndex( self, new_action, index ):
        """ Must be overridden by subclasses.
        Set the action for this index to new_action, then either call this method
        or call setDirty() directly to mark this file as edited.
        e.g. self.actions[index] = new_action; self.setDirty()"""

        self.setDirty()

    def deleteIndex( self, index ):
        """ May be overridden by subclasses.
        Delete all data associated with the sample at index.
        Call setDirty() to mark this file as edited. """

        pass

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

    def supportsAuxData(self):
        """ Does this class support adding auxiliary data types? """
        return False

    @staticmethod
    def defaultInputTypes():
        """ Return an array of dicts describing the input types.
        e.g. [{"name":"Images", "type":"numpy image", "shape":(120,120,3)}] """
        return [{}]

    def inputTypes(self):
        return DriveFormat.defaultInputTypes()

    @staticmethod
    def defaultOutputTypes():
        """  Return an array of dicts describing the output types.
        e.g. [{"name":"Actions", "type":"categorical", "categories":[ "forward", "backward", "left", "right", "stop" ]}] """
        return [{}]

    def outputTypes(self):
        return DriveFormat.defaultOutputTypes()

    @staticmethod
    def testFormat( FormatClass, test_path, invalid_action ):

        d = FormatClass(test_path)

        print( "Testing {}".format( FormatClass ) )
        print( "Meta data:\n{}".format( d.meta ) )
        print( "Image 10 shape: {}".format( d.imageForIndex(9).shape ) )
        print( "Action 10: {}".format( d.actionForIndex(9) ) )
        print( "Actions length: {} ".format( len(d.actions) ) )
        if d.isClean():
            print( "Drive is clean before edit PASS" )
        else:
            print( "Drive is dirty before edit: FAIL" )

        invalid_action = 'very long action'
        before = d.actionForIndex(3)
        d.setActionForIndex(invalid_action,3)
        after = d.actionForIndex(3)
        if invalid_action != after:
            print( "Set action before/set/after: {}/{}/{}: FAIL".format( before, invalid_action, after ) )
        else:
            print( "Set action succeeded: PASS" )
        if d.isClean():
            print( "Drive is clean after edit: FAIL" )
        else:
            print( "Drive is dirty after edit: PASS" )

        try:
            d = FormatClass("DriveFormat.py")
        except IOError as ex:
            print( "Caught correct exception when path is not a directory: PASS" )
        except Exception as exg:
            print( "Caught invalid exception ({}) when path is not a directory: FAIL".format(exg) )
        else:
            print( "No exception raised when path is not a directory: FAIL" )

        try:
            d = FormatClass("NonExistantDrivePath_________")
        except IOError as ex:
            print( "Caught correct exception when path does not exist: PASS" )
        except Exception as exg:
            print( "Caught invalid exception ({}) when path does not exist: FAIL".format(exg) )
        else:
            print( "No exception raised when path does not exist: FAIL" )

        handler = DriveFormat.handlerForFile( test_path )
        if handler is None:
            print( "Failed to find class for test_path: FAIL" )
        elif not isinstance(handler,FormatClass):
            print( "Found wrong class for test_path: FAIL" )
        else:
            print( "Found correct class for test_path: PASS" )

        handler = DriveFormat.handlerForFile( "DriveFormat.py" )
        if handler is not None:
            print( "Found class for invalid file type: Fail" )
        else:
            print( "Found no class for invalid file type: PASS" )

        try:
            DriveFormat.registerFormat( "test_format", FormatClass )
        except Exception as exg:
            print( "Caught invalid exception ({}) when registering format: FAIL".format(exg) )
        else:
            print( "No exception when registering format: PASS" )

        try:
            DriveFormat.registerFormat( "test_format", FormatClass )
        except Exception as exg:
            print( "Caught invalid exception ({}) when re-registering format: FAIL".format(exg) )
        else:
            print( "No exception when re-registering format: PASS" )

        class FakeDrive:
            pass

        try:
            DriveFormat.registerFormat( "test_format", FakeDrive )
        except ValueError as exr:
            print( "Caught correct exception when registering a duplicate format: PASS" )
        except Exception as exg:
            print( "Caught invalid exception ({}) when registering a duplicate format: FAIL".format(exg) )
        else:
            print( "No exception when registering a duplicate format: FAIL" )


        if FormatClass.canOpenFile( test_path ):
            print( "Can open file {}: PASS".format(test_path) )
        else:
            print( "Can't open file {}: FAIL".format(test_path) )
