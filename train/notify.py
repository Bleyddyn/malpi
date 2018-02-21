#! /usr/bin/python

import sys
from optparse import OptionParser, OptionGroup, IndentedHelpFormatter

# Import smtplib to provide email functions
import smtplib
 
# Import the email modules
from email.mime.text import MIMEText

try:
    import config
except:
    print "Failed to load config file config.py."
    print "Try copying config_empty.py to config.py, fill in the details, and re-run."
    exit()
 
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

def _parse_cmdline():
    usage = "mail.py --subject <subject> --message <message> <email(s)>"

    parser = OptionParser( usage=usage, description="Send an email using my Gmail account" )
    parser.add_option( '--subject', help="Subject for the email" );
    parser.add_option( '--message', help="Message body. If none, expect body on standard input" );

    parser.disable_interspersed_args()
    (options, args) = parser.parse_args()

    if len(args) == 0:
        print "At least one destination email address is required."
        parser.print_help()
        sys.exit(1)

    return (options, args)


if __name__ == "__main__":
    (options, args) = _parse_cmdline()

    if options.message:
        msg = options.message
    else:
        msg = sys.stdin.read()

    mailTo( args[0], subject=options.subject, message=msg )
