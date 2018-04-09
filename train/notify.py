#! /usr/bin/python

import sys
import argparse

# Import smtplib to provide email functions
import smtplib
 
# Import the email modules
from email.mime.text import MIMEText

try:
    import config
except:
    print( "Failed to load config file config.py." )
    print( "Try copying config_empty.py to config.py, fill in the details, and re-run." )
    exit()

try:
    import mac_notify
    mac_notifications = True
except:
    mac_notifications = False
    print( "Failed to import mac_notify. Mac OS X Notifications will not be sent." )

def mailTo( addr_to, subject='Notification', message='' ):
    """ Send an email using an smtp account.
      Email addresses (comma delimited if more than one)
      Optional subject and message body
    """

# Define email addresses to use
    addr_from = config.notifications['addr_from']

# Define SMTP email server details
    smtp_server = config.notifications['smtp_server']
    smtp_user   = config.notifications['smtp_user']
    smtp_pass   = config.notifications['smtp_pass']

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

def notify( title, subTitle='', message='', email_to=None, mac=True, sound=False ):
    if email_to is not None:
        if subTitle is None:
            msg2 = message
        else:
            msg2 = subTitle + "\n" + message
        mailTo( email_to, subject=title, message=msg2 )
    if mac and mac_notifications:
        mac_notify.notify( title, subTitle, message, sound=sound )

def _parse_cmdline():
    parser = argparse.ArgumentParser(description='Send a notification by email and/or Mac OS X notification system .', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( 'title', nargs=1, help="Title for the notification and/or Subject for the email" );
    parser.add_argument( '--message', help="Message body. If none, expect body on standard input" );
    parser.add_argument( '--sub', help="Subtitle." );
    parser.add_argument( '--email', help="Email address to send the notification to");
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mac', action="store_true", default=True, help='Send a Mac OS X Notification')
    group.add_argument('--no_mac', action="store_false", dest='mac', default=False, help="Don't Send a Mac OS X Notification")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _parse_cmdline()

    if args.message:
        msg = args.message
    else:
        msg = sys.stdin.read()

    notify( args.title[0], subTitle=args.sub, message=msg, email_to=args.email, mac=args.mac )
