#! /usr/bin/python

import sys
import argparse
import json

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

def read_email_config( fname="email_config.json" ):
    """ Read the json config file and return notification related entries.
    """
    try:
        with open(fname,'r') as f:
            notifications = json.load(f)
    except Exception as ex:
        print( "Failed to load email config file {}".format( fname ) )
        raise

    all_keys = {'addr_from', 'smtp_pass', 'smtp_server', 'smtp_user'}
    read_keys = set(notifications)
    if not all_keys.issubset( read_keys ):
        raise KeyError("Key(s) missing from email notifications config file: {}".format( ", ".join( list(all_keys.difference(read_keys)) )))

    return notifications

# TODO Add a write_email_config function

def notify( title, subTitle='', message='', email_to=None, mac=True, sound=False, email_config={} ):
    if email_to is not None:
        if subTitle is None:
            msg2 = message
        else:
            msg2 = subTitle + "\n" + message
        mailTo( email_to, subject=title, message=msg2, config=email_config )
    if mac and mac_notifications:
        mac_notify( title, subTitle, message, sound=sound )
