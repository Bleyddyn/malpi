from PyObjCTools import AppHelper
import Foundation
import objc
import AppKit

# objc.setVerbose(1)

#def simpler():
#    import Foundation
#    import objc
#    import AppKit
#    import sys
#
#    NSUserNotification = objc.lookUpClass('NSUserNotification')
#    NSUserNotificationCenter = objc.lookUpClass('NSUserNotificationCenter')
#
#    def notify(title, subtitle, info_text, delay=0, sound=False, userInfo={}):
#        notification = NSUserNotification.alloc().init()
#        notification.setTitle_(title)
#        notification.setSubtitle_(subtitle)
#        notification.setInformativeText_(info_text)
#        notification.setUserInfo_(userInfo)
#        if sound:
#            notification.setSoundName_("NSUserNotificationDefaultSoundName")
#        notification.setDeliveryDate_(Foundation.NSDate.dateWithTimeInterval_sinceDate_(delay, Foundation.NSDate.date()))
#        NSUserNotificationCenter.defaultUserNotificationCenter().scheduleNotification_(notification)
#
#
#    notify("Test message", "Subtitle", "This message should appear instantly, with a sound", sound=True)
#    sys.stdout.write("Notification sent...\n")

class MountainLionNotification(Foundation.NSObject):
    # Based on http://stackoverflow.com/questions/12202983/working-with-mountain-lions-notification-center-using-pyobjc

    def init(self):
        self = super(MountainLionNotification, self).init()
        if self is None: return None

        # Get objc references to the classes we need.
        self.NSUserNotification = objc.lookUpClass('NSUserNotification')
        self.NSUserNotificationCenter = objc.lookUpClass('NSUserNotificationCenter')

        return self

    def clearNotifications(self):
        """Clear any displayed alerts we have posted. Requires Mavericks."""

        NSUserNotificationCenter = objc.lookUpClass('NSUserNotificationCenter')
        NSUserNotificationCenter.defaultUserNotificationCenter().removeAllDeliveredNotifications()

    @objc.python_method
    def notify(self, title, subtitle, text, url=None, sound=False):
        """Create a user notification and display it."""

        notification = self.NSUserNotification.alloc().init()
        notification.setTitle_(str(title))
        if subtitle is not None:
            notification.setSubtitle_(str(subtitle))
        notification.setInformativeText_(str(text))
        notification.setSoundName_("NSUserNotificationDefaultSoundName")
        notification.setHasActionButton_(True)
        notification.setActionButtonTitle_("View")
        if sound:
            notification.setSoundName_("NSUserNotificationDefaultSoundName")
        if url is not None:
            notification.setUserInfo_({"action":"open_url", "value":url})

        self.NSUserNotificationCenter.defaultUserNotificationCenter().setDelegate_(self)
        self.NSUserNotificationCenter.defaultUserNotificationCenter().scheduleNotification_(notification)

        # Note that the notification center saves a *copy* of our object.
        return notification

    def userNotificationCenter_didActivateNotification_(self, center, notification):
        """Handle when a user clicks on one of our posted notifications.
           This will only work if self still exists at the time of the user interaction.
           This has not worked in testing."""

        userInfo = notification.userInfo()
        if userInfo["action"] == "open_url":
            import subprocess
            subprocess.Popen(['open', userInfo["value"]])

def notify( title, subTitle, message, sound=False ):
    mnot = MountainLionNotification.alloc().init()
    mnot.notify( title, subTitle, message, sound=sound )

    # In case the caller needs to keep the object around to handle user interaction
    return mnot
