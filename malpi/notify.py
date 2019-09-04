#! /usr/bin/python

import sys
import argparse

# Import smtplib to provide email functions
import smtplib
 
# Import the email modules
from email.mime.text import MIMEText

try:
    from .mac_notify import notify as mac_notify
    mac_notifications = True
except Exception as ex:
    mac_notifications = False
    print( "Failed to import mac_notify. Mac OS X Notifications will not be sent." )
    print( "Exception: {}".format( ex ) )

def mailTo( addr_to, subject='Notification', message='', config={} ):
    """ Send an email using an smtp account.
      Email addresses (comma delimited if more than one)
      Optional subject and message body
      @config requires the following keys: addr_from, smtp_server, smtp_user, smtp_pass
    """

# Define email addresses to use
    addr_from = config['addr_from']

# Define SMTP email server details
    smtp_server = config['smtp_server']
    smtp_user   = config['smtp_user']
    smtp_pass   = config['smtp_pass']

    if not subject:
        subject = 'Notification'

# Construct email
    msg = MIMEText(message)
    msg['To'] = addr_to
    msg['From'] = addr_from
    msg['Subject'] = subject

# Send the message via an SMTP server
    s = smtplib.SMTP_SSL()
    s.connect(smtp_server, 465)
    s.login(smtp_user,smtp_pass)
    s.sendmail(addr_from, addr_to, msg.as_string())
    s.quit()

def get_email_config():
    """ Read the MaLPi style config.py file and return notification related entries.
    """
    try:
        import config
    except Exception as ex:
        print( "Failed to load config file config.py." )
        print( "Try copying config_empty.py to config.py, fill in the details, and re-run." )
        raise

    return config.notifications

def notify( title, subTitle='', message='', email_to=None, mac=True, sound=False, email_config={} ):
    if email_to is not None:
        if subTitle is None:
            msg2 = message
        else:
            msg2 = subTitle + "\n" + message
        mailTo( email_to, subject=title, message=msg2, config=email_config )
    if mac and mac_notifications:
        mac_notify( title, subTitle, message, sound=sound )
